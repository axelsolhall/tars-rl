# TARS roadmap

Agreed 2026-11-02.

## The goal

TARS walks around a real apartment, maps it, and finds things you ask for.
- An LLM decider sits on top and places flags.
- An RL motion policy below walks to them.
- The policy is trained in simulation and moved to the real robot (sim-to-real).

## The rule that stops polishing

Every stage has a pass bar and a timebox. When the time runs out, move on with what you have.

Every stage is verified somewhere harder than where it was built: MuJoCo, then Unreal, then reality. This follows the
build order from 2026-10-18: build a crude but complete end-to-end version first, then improve the weakest stage.

## Stages

| # | Stage | Pass bar ("done when") | Verified in |
|---|---|---|---|
| 1 | Memory (compare encoders over the last 0.5 s of states) | best memory type chosen | MuJoCo, held-out flags |
| 2 | Domain randomization (`tars_rl.env.randomize`) | flags per episode hold up on randomized TARS | MuJoCo, randomized, held-out |
| 3 | Rough-terrain curriculum (see below) | walks the hardest level without many falls | MuJoCo |
| 4 | Unreal gate, the hard check (see below) | flags/ep and falls close to MuJoCo's | Unreal |
| 5 | Commands: stand still, turn on the spot, plus the flag | the decider can steer it reliably | Unreal |
| 6 | Decider in Unreal (see below) | "find the toy" works within N seconds | Unreal |
| 7 | Real TARS (see below) | real TARS walks to a point on the floor | reality |
| 8 | Real decider and mapping: a camera loop and a simple map of visited places | "find the toy" in the apartment | reality |

What the longer stages contain:
- **Stage 3, rough terrain:**
  - uneven floor, ±1–2 cm;
  - small steps, 1–3 cm (thresholds, rugs);
  - gentle slopes.
  The terrain gets harder as TARS succeeds.
- **Stage 4, Unreal gate:** the policy trained in MuJoCo runs in Unreal's own physics, on uneven floors and steps.
- **Stage 6, decider in Unreal:** an apartment-like level with objects. The loop is camera → Claude → flag.
- **Stage 7, real TARS:**
  - print the parts;
  - fit servos, an IMU and a small computer;
  - measure the real servos and update the simulator to match.

## Design choices

**The motion policy stays blind.**
- It feels the floor only through its own body, which makes it robust to small bumps and steps.
- The decider handles obstacles: it sees them and routes flags around them.
- This is how state-of-the-art hierarchies split the work. It also keeps a depth camera out of the RL loop.

**Unreal is a sim-to-sim test before sim-to-real.**
- A different physics engine is the cheapest honest preview of the reality gap.
- A policy that only works in MuJoCo fails in Unreal, not on the floor.

## Timing

- Stages 1–4 should take a few weeks.
- Stage 7 depends on hardware lead time, so start it in parallel now:
  - the printer;
  - the TARS part designs;
  - servos and an IMU.
- Doing hardware in parallel also guards against polishing: reality keeps pulling the project forward.
