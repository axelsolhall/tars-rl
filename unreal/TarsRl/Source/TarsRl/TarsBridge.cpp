#include "TarsBridge.h"

#include "Camera/CameraActor.h"
#include "Common/TcpSocketBuilder.h"
#include "Dom/JsonObject.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "HAL/PlatformProcess.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"
#include "PhysicsEngine/PhysicsSettings.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "TarsActor.h"
#include "TarsLevel.h"

DEFINE_LOG_CATEGORY_STATIC(LogTarsBridge, Log, All);

namespace
{
constexpr int32 Protocol = 1;  // tars_rl.unreal.bridge.PROTOCOL
constexpr double ClientTimeoutS = 120.0;  // a silent client this long is gone

TArray<TSharedPtr<FJsonValue>> Numbers(const FVector& V)
{
	return {MakeShared<FJsonValueNumber>(V.X), MakeShared<FJsonValueNumber>(V.Y), MakeShared<FJsonValueNumber>(V.Z)};
}

TArray<TSharedPtr<FJsonValue>> Numbers(const FQuat& Q)
{
	return {MakeShared<FJsonValueNumber>(Q.X), MakeShared<FJsonValueNumber>(Q.Y), MakeShared<FJsonValueNumber>(Q.Z),
		MakeShared<FJsonValueNumber>(Q.W)};
}

bool ReadVec(const TSharedPtr<FJsonObject>& O, const TCHAR* Key, FVector& Out)
{
	const TArray<TSharedPtr<FJsonValue>>* A = nullptr;
	if (!O->TryGetArrayField(Key, A) || A->Num() != 3) return false;
	Out = FVector((*A)[0]->AsNumber(), (*A)[1]->AsNumber(), (*A)[2]->AsNumber());
	return true;
}

int32 Substeps(double Dt)
{
	const UPhysicsSettings* S = UPhysicsSettings::Get();
	if (!S->bSubstepping) return 1;
	return FMath::Min(FMath::CeilToInt(Dt / S->MaxSubstepDeltaTime - 1e-6), S->MaxSubsteps);
}
}  // namespace

ATarsBridge::ATarsBridge()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PrePhysics;  // targets must be set before this frame's physics
	PrimaryActorTick.bTickEvenWhenPaused = true;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

void ATarsBridge::BeginPlay()
{
	Super::BeginPlay();
	FIPv4Endpoint Endpoint(FIPv4Address(127, 0, 0, 1), Port);
	Listener = FTcpSocketBuilder(TEXT("TarsBridge")).AsReusable().AsNonBlocking().BoundToEndpoint(Endpoint).Listening(1);
	if (!Listener)
	{
		UE_LOG(LogTarsBridge, Error, TEXT("cannot listen on 127.0.0.1:%d (another Unreal on that port?)"), Port);
		return;
	}
	UE_LOG(LogTarsBridge, Log, TEXT("TARS bridge listening on 127.0.0.1:%d, %.3f s steps, %d physics substeps"), Port,
		ControlDt, Substeps(ControlDt));

	FActorSpawnParameters P;
	P.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
	Level = GetWorld()->SpawnActor<ATarsLevel>(Origin, FRotator::ZeroRotator, P);
	Camera = GetWorld()->SpawnActor<ACameraActor>(Origin + FVector(-150, 150, 90), FRotator::ZeroRotator, P);
	if (APlayerController* PC = GetWorld()->GetFirstPlayerController()) PC->SetViewTarget(Camera);
}

void ATarsBridge::EndPlay(const EEndPlayReason::Type Reason)
{
	ISocketSubsystem* Sockets = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	if (Client) Sockets->DestroySocket(Client);
	if (Listener) Sockets->DestroySocket(Listener);
	Client = Listener = nullptr;
	Super::EndPlay(Reason);
}

void ATarsBridge::Accept()
{
	bool bPending = false;
	if (!Listener || !Listener->HasPendingConnection(bPending) || !bPending) return;
	Client = Listener->Accept(TEXT("TarsBridgeClient"));
	if (!Client) return;
	Client->SetNonBlocking(false);
	Client->SetNoDelay(true);
	Pending.Reset();
	bStepPending = false;
	UE_LOG(LogTarsBridge, Log, TEXT("Python connected"));
}

void ATarsBridge::DropClient(const FString& Why)
{
	UE_LOG(LogTarsBridge, Log, TEXT("Python disconnected: %s"), *Why);
	if (Client) ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Client);
	Client = nullptr;
	bStepPending = false;
	Hud = TEXT("waiting for Python");
}

bool ATarsBridge::ReadLine(FString& Line, double TimeoutS)
{
	const double Deadline = FPlatformTime::Seconds() + TimeoutS;
	while (true)
	{
		const int32 Nl = Pending.Find('\n');
		if (Nl != INDEX_NONE)
		{
			// The line is bytes [0, Nl); decode exactly those.
			const FUTF8ToTCHAR Conv(reinterpret_cast<const ANSICHAR*>(Pending.GetData()), Nl);
			Line = FString(Conv.Length(), Conv.Get());
			Pending.RemoveAt(0, Nl + 1);
			return true;
		}
		const double Left = Deadline - FPlatformTime::Seconds();
		if (Left <= 0 || !Client->Wait(ESocketWaitConditions::WaitForRead, FTimespan::FromSeconds(Left))) return false;
		uint8 Buf[65536];
		int32 Read = 0;
		if (!Client->Recv(Buf, sizeof(Buf), Read) || Read <= 0) return false;
		Pending.Append(Buf, Read);
	}
}

void ATarsBridge::Send(const TSharedRef<FJsonObject>& Msg)
{
	FString Text;
	const TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> W =
		TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Text);
	FJsonSerializer::Serialize(Msg, W);
	Text += TEXT("\n");
	const FTCHARToUTF8 Utf8(*Text);
	int32 Done = 0;
	while (Client && Done < Utf8.Length())
	{
		int32 Sent = 0;
		if (!Client->Send(reinterpret_cast<const uint8*>(Utf8.Get()) + Done, Utf8.Length() - Done, Sent))
		{
			DropClient(TEXT("send failed"));
			return;
		}
		Done += Sent;
	}
}

void ATarsBridge::SendError(const FString& Error)
{
	TSharedRef<FJsonObject> M = MakeShared<FJsonObject>();
	M->SetStringField(TEXT("error"), Error);
	Send(M);
}

TSharedRef<FJsonObject> ATarsBridge::StateMessage(float FrameDt) const
{
	TSharedRef<FJsonObject> M = MakeShared<FJsonObject>();
	M->SetNumberField(TEXT("t"), EpisodeTime);
	M->SetNumberField(TEXT("dt"), FrameDt);
	TArray<TSharedPtr<FJsonValue>> Bodies;
	for (const FTarsSlabState& S : Tars->ReadSlabs())
	{
		TSharedRef<FJsonObject> B = MakeShared<FJsonObject>();
		B->SetArrayField(TEXT("p"), Numbers(S.Pos - Origin));
		B->SetArrayField(TEXT("q"), Numbers(S.Rot));
		B->SetArrayField(TEXT("v"), Numbers(S.Vel));
		B->SetArrayField(TEXT("w"), Numbers(S.AngVel));
		Bodies.Add(MakeShared<FJsonValueObject>(B));
	}
	M->SetArrayField(TEXT("bodies"), Bodies);
	return M;
}

bool ATarsBridge::Handle(const TSharedPtr<FJsonObject>& Cmd)
{
	const FString Op = Cmd->GetStringField(TEXT("op"));
	TSharedRef<FJsonObject> Ok = MakeShared<FJsonObject>();
	Ok->SetBoolField(TEXT("ok"), true);
	if (Op == TEXT("hello"))
	{
		Ok->SetNumberField(TEXT("protocol"), Protocol);
		Ok->SetNumberField(TEXT("dt"), ControlDt);
		Ok->SetNumberField(TEXT("substeps"), Substeps(ControlDt));
		Send(Ok);
	}
	else if (Op == TEXT("level"))
	{
		FString Error;
		const int32 N = Level->Build(Cmd->GetArrayField(TEXT("boxes")), Error);
		if (N < 0) SendError(Error);
		else
		{
			Ok->SetNumberField(TEXT("boxes"), N);
			Send(Ok);
		}
	}
	else if (Op == TEXT("reset"))
	{
		FVector At, Flag;
		if (!ReadVec(Cmd, TEXT("at"), At) || !ReadVec(Cmd, TEXT("flag"), Flag))
		{
			SendError(TEXT("reset needs at and flag, 3 numbers each"));
			return false;
		}
		if (Tars) Tars->Destroy();
		const FTransform Where(FRotator(0, Cmd->GetNumberField(TEXT("yaw")), 0), Origin + At);
		Tars = GetWorld()->SpawnActorDeferred<ATarsActor>(ATarsActor::StaticClass(), Where, nullptr, nullptr,
			ESpawnActorCollisionHandlingMethod::AlwaysSpawn);
		Tars->bScripted = false;
		Tars->FinishSpawning(Where);
		if (!Tars->IsReady())
		{
			SendError(TEXT("TARS failed to build: body_spec.json missing or wrong (python -m tars_rl.unreal.export)"));
			return false;
		}
		BodyHeightCm = Tars->BodyHeightCm();
		Level->SetFlag(Flag, BodyHeightCm);
		EpisodeTime = 0;
		Send(StateMessage(0.f));
	}
	else if (Op == TEXT("flag"))
	{
		FVector Pos;
		if (!ReadVec(Cmd, TEXT("pos"), Pos))
		{
			SendError(TEXT("flag needs pos, 3 numbers"));
			return false;
		}
		Level->SetFlag(Pos, BodyHeightCm);
		Send(Ok);
	}
	else if (Op == TEXT("step"))
	{
		const TArray<TSharedPtr<FJsonValue>>* Ctrl = nullptr;
		if (!Tars || !Cmd->TryGetArrayField(TEXT("ctrl"), Ctrl) || Ctrl->Num() != 4)
		{
			SendError(Tars ? TEXT("step needs ctrl, 4 servo targets") : TEXT("step before reset"));
			return false;
		}
		Tars->SetTargets({(*Ctrl)[0]->AsNumber(), (*Ctrl)[1]->AsNumber(), (*Ctrl)[2]->AsNumber(), (*Ctrl)[3]->AsNumber()});
		Cmd->TryGetStringField(TEXT("hud"), Hud);
		return true;
	}
	else if (Op == TEXT("bye"))
	{
		Send(Ok);
		DropClient(TEXT("bye"));
	}
	else
	{
		SendError(FString::Printf(TEXT("unknown op '%s'"), *Op));
	}
	return false;
}

void ATarsBridge::FollowCamera(float DeltaSeconds)
{
	if (!Camera) return;
	// Possessing a pawn resets the view target; keep ours (cheap: one compare per frame).
	APlayerController* PC = GetWorld()->GetFirstPlayerController();
	if (PC && PC->GetViewTarget() != Camera) PC->SetViewTarget(Camera);
	if (!Tars) return;
	const FVector Target = Tars->CentralLocation();
	const FVector Want = Target + FVector(-160, 160, 100);  // behind-left of TARS's start heading
	const float A = FMath::Clamp(DeltaSeconds * 2.f, 0.f, 1.f);
	Camera->SetActorLocation(FMath::Lerp(Camera->GetActorLocation(), Want, A));
	Camera->SetActorRotation((Target - Camera->GetActorLocation()).Rotation());
}

void ATarsBridge::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (!Client) Accept();
	if (!Client)
	{
		if (GEngine) GEngine->AddOnScreenDebugMessage(10, 1.f, FColor::Yellow,
			FString::Printf(TEXT("TARS bridge: waiting for Python on port %d"), Port));
		FPlatformProcess::Sleep(static_cast<float>(ControlDt));  // fixed steps never wait: idle at ~real time
		FollowCamera(DeltaSeconds);
		return;
	}
	if (bStepPending)
	{
		EpisodeTime += DeltaSeconds;
		Send(StateMessage(DeltaSeconds));  // the state the last frame's physics produced
		bStepPending = false;
	}
	FollowCamera(static_cast<float>(ControlDt));
	if (GEngine && !Hud.IsEmpty()) GEngine->AddOnScreenDebugMessage(10, 1.f, FColor::White, Hud);
	FString Line;
	while (Client)
	{
		if (!ReadLine(Line, ClientTimeoutS))
		{
			DropClient(TEXT("closed or silent"));
			return;
		}
		TSharedPtr<FJsonObject> Cmd;
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Line), Cmd) || !Cmd.IsValid() ||
			!Cmd->HasField(TEXT("op")))
		{
			SendError(TEXT("each line must be a JSON object with an op"));
			continue;
		}
		if (Handle(Cmd))
		{
			bStepPending = true;  // physics runs now; the reply goes out at the next frame's start
			return;
		}
	}
}
