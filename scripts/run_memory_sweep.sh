#!/usr/bin/env bash
# Reference memory mu = N / m against the class-block duration (Table 7, Figures 6-7).
# N = windows per forward pass (--batch), m = update rate (--momentum).
set -e
cd "$(dirname "$0")/.."
# offline (reported update order)
python train.py --config config/cohortA.yaml --batch 8 --out a_batch8_3seed.json
python train.py --config config/cohortA.yaml --arms stem_tangent --batch 8 --momentum 1.0 --out a_batch8_m1_3seed.json
python train.py --config config/cohortB.yaml --batch 256 --out b_batch256_3seed.json
python train.py --config config/cohortB.yaml --arms stem_tangent --momentum 0.01 --out b_tangent_m001_3seed.json
for n in 8 4 2; do
  python train.py --config config/cohortC.yaml --batch $n --momentum 1.0 --out c_boundary_mu$n.json
done
python train.py --config config/cohortC.yaml --batch 4 --momentum 0.25  --out c_fixedN_N4_m0.25.json
python train.py --config config/cohortC.yaml --batch 2 --momentum 0.125 --out c_fixedN_N2_m0.125.json

# fully causal: "cohort letter N m" -> <c>_fullcausal_mu<N/m>.json
FC="--causal-filter --causal-update --arms stem_tangent"
while read c n m; do
  mu=$(python -c "print(round($n/$m))")
  python train.py --config config/cohort${c^^}.yaml $FC --batch $n --momentum $m --out ${c}_fullcausal_mu$mu.json
done <<'EOF'
a 8 1.0
a 32 1.0
a 32 0.05
c 32 1.0
c 32 0.05
c 8 1.0
c 4 1.0
c 2 1.0
d 32 1.0
d 32 0.05
e 8 1.0
e 32 1.0
e 32 0.05
b 32 0.01
EOF
# cohort C controls at mu = 4: the same memory with the reported update order,
# and N = 4 with a long memory
python train.py --config config/cohortC.yaml --causal-filter --arms stem_tangent --batch 4 --momentum 1.0 --out c_fullcausal_mu4_noncausalupd.json
python train.py --config config/cohortC.yaml $FC --batch 4 --momentum 0.025 --out c_fullcausal_N4_mu160.json
