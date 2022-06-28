# how to replicate experiments
slurm scripts call the py scripts in this folder, usually of the same name.
Each slurm script has their own command line parameters. Please refer to the table below for individual settings for each experiment (these have to be manually changed in the slurm scripts).

# experimental settings
BASEL   DATASET     EPOCHS     TIME     PARTITION
RCAN    CERRADO     60         2h       GPU-SHORT
        UC_MERCED   100        24h      GPU-MEDIUM
        OLI2MSI     50         24h      GPU-MEDIUM
        SO2SAT      18         120      GPU-LONG
        BIGEARTHNET
        SENT-NICFI
WDSR    CERRADO     60         2h       GPU-SHORT
        UC_MERCED   100        24h      GPU-MEDIUM
        OLI2MSI     50         24h      GPU-MEDIUM
        SO2SAT      18         120      GPU-LONG
        BIGEARTHNET
        SENT-NICFI

AUTOKERAS       DATASET         TRIALS  TIME    PARTITION
CNN             CERRADO         10      2:00    GPU-SHORT
                CERRADO         10      4:00    GPU-SHORT
                UC_MERCED       10      24:00   GPU-MEDIUM
                UC_MERCED       10      48:00   GPU-LONG 
                OLI2MSI         10      48:00   GPU-LONG
                0LI2MSI         10      96:00   GPU-LONG
                SO2SAT          10      48:00   GPU-LONG      
                SO2SAT          10      96:00   GPU-LONG      
                SENT_NICFI      10      4:00    GPU-SHORT
                SENT_NICFI      10      8:00    GPU-MEDIUM