# Description of each script

base.py: recreation of Ahuja et al. 2026's article about surface temp and amb temp --> resistance based soh

voltage_with_ecm.py: calibration of variables needed for 1st order thevenin model for V, V(t) = OCV(SoC(t)) - I*\R0(t) - I*\Rp(t)*(1-e^(-t/tau)) \
focuses on R0/Rp split, estimates R_total as a function of SoC, OCV (not V(t)) curve

full_extended_mhe.py: full mhe that ties step 1 and 2 together \
usage tutorial at the top of the file
