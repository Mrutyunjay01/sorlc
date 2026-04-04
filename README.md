# SORLC (Scaling OpenEnv RL in Chess)

Idea is to learn the intuitions and built instinct for scaling RL. After procastinating for eons and juggling work, finally boarding the ship of RL and frontier AI (hopefully), thanks for the [Meta OpenEnv India Hackathon](https://pytorch.org/event/openenv-ai-hackathon/). Agenda of this repo would be to start from a basic chess engine with min-max pruning may be, then to move to stockfish-like engine, then scale it using open-env. (Yea, I am GPU-poor and tokens-poor!). Kindly bear with the messy, unpolished earlier commits.

## Rough plan

- [x] chess engine with player
- [x] basic rl agent to play and improve (min-max, or something)
- [x] think about abstraction here: engine as server environment, rl agent as client
- [ ] move to open-env
- [ ] might want to merge `StepResult` with `Observation`, and introduce `State` -> to align with classical RL paradigm.
- [ ] scale open-env from one client-server to multiple to run and evaluate multiple strategies
- [ ] automate scaling (spawning) servers as a k8s task (like kubernete meets open-env, not sure if this has been attempted yet from open-env context at all, can do a search and confirm)
