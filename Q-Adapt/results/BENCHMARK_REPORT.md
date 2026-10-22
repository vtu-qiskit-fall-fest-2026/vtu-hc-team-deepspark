# Q-Adapt benchmark output

shots/circuit/trial=5000, trials=6, seed=2026, budget=5, lambda=0.005, alpha=0.05

qiskit 2.6.0, qiskit-aer 0.18.0, python 3.13.16

All noise is SIMULATED. Adaptive rows use held-out validation trials.


## bitflip_sweep  (task=quantum)

Adaptive choices:

```
   p_x adaptive_choice  heldout_failure_rate  heldout_baseline_rate  confident
0.0000            none                0.0000                 0.0000      False
0.0250          bit3_q                0.0009                 0.0130       True
0.0500          bit3_q                0.0030                 0.0257       True
0.0750          bit3_q                0.0081                 0.0366       True
0.1000          bit3_q                0.0139                 0.0515       True
0.1250          bit5_q                0.0087                 0.0624       True
0.1500          bit5_q                0.0125                 0.0740       True
0.1750          bit5_q                0.0206                 0.0882       True
0.2000          bit5_q                0.0282                 0.0980       True
0.2250          bit5_q                0.0401                 0.1128       True
0.2500          bit5_q                0.0515                 0.1235       True
```


## phaseflip_sweep  (task=quantum)

Adaptive choices:

```
   p_z adaptive_choice  heldout_failure_rate  heldout_baseline_rate  confident
0.0000            none                0.0000                 0.0000      False
0.0250            none                0.0128                 0.0128       True
0.0500        phase3_q                0.0033                 0.0240       True
0.0750        phase3_q                0.0084                 0.0360       True
0.1000        phase3_q                0.0132                 0.0487       True
0.1250        phase5_q                0.0079                 0.0660       True
0.1500        phase5_q                0.0129                 0.0721       True
0.1750        phase5_q                0.0195                 0.0863       True
0.2000        phase5_q                0.0281                 0.1004       True
0.2250        phase5_q                0.0401                 0.1132       True
0.2500        phase5_q                0.0508                 0.1239       True
```


## gate_noise_sweep_px0.05  (task=quantum)

Adaptive choices:

```
 p_gate adaptive_choice  heldout_failure_rate  heldout_baseline_rate  confident
 0.0000          bit3_q                0.0033                 0.0255       True
 0.0060            none                0.0267                 0.0267       True
 0.0120            none                0.0257                 0.0257       True
 0.0180            none                0.0250                 0.0250       True
 0.0240            none                0.0260                 0.0260       True
 0.0300            none                0.0246                 0.0246       True
 0.0360            none                0.0240                 0.0240       True
 0.0420            none                0.0244                 0.0244       True
 0.0480            none                0.0243                 0.0243       True
 0.0540            none                0.0254                 0.0254       True
 0.0600            none                0.0249                 0.0249       True
```


## classical_readout_sweep_px0.05  (task=classical)

Adaptive choices:

```
 p_meas adaptive_choice  heldout_failure_rate  heldout_baseline_rate  confident
 0.0000          bit3_q                0.0033                 0.0255       True
 0.0250          bit5_v                0.0027                 0.0511       True
 0.0500          bit5_v                0.0080                 0.0734       True
 0.0750          bit5_v                0.0133                 0.0970       True
 0.1000          bit5_v                0.0209                 0.1181       True
 0.1250          bit5_v                0.0321                 0.1449       True
 0.1500          bit5_v                0.0470                 0.1693       True
 0.1750          bit5_v                0.0650                 0.1888       True
 0.2000          bit5_v                0.0835                 0.2161       True
 0.2250          bit5_v                0.1040                 0.2398       True
 0.2500          bit5_v                0.1303                 0.2650       True
```


## Regime map (rows p_z, columns p_x), budget 5

```
p_x       0.00    0.04    0.08    0.12    0.16    0.20
p_z                                                   
0.20  phase5_q    none    none    none    none    none
0.16  phase5_q    none    none    none    none    none
0.12  phase3_q    none    none    none    none    none
0.08  phase3_q    none    none    none    none    none
0.04  phase3_q    none    none    none    none    none
0.00      none  bit3_q  bit3_q  bit5_q  bit5_q  bit5_q
```
