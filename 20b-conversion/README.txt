1. Download the 46 pytorch_model-#####-of-00046.bin files from https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main into the "in" folder.
2. Run phase1.py. This creates another 40 GB worth of files in the phase1_out folder.
3. If you are low on disk space, you can delete the contents of the in folder.
4. Run phase2.py. This creates another 40 GB worth of files in the phase2_out folder.
5. Upload the files in the phase2_out folder to your Google Cloud Storage.

The scripts take very little system memory and can be run on a computer with 4 GB of memory and no GPU.
