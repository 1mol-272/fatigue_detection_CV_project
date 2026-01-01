# videos/ (Input)

This project uses the **UTA Real-Life Drowsiness Dataset (UTA-RLDD)**.

Dataset page (download link included):
https://sites.google.com/view/utarldd/home

> Note: The full dataset is very large (~111 GB). Do **NOT** commit videos to Git.

## 1. Labels rule

UTA-RLDD provides three classes, encoded as numeric labels:
- 0 = alert
- 5 = low vigilant
- 10 = drowsy

## 2. What we use in this repo
We only use **label 0** and **label 10** videos.
Label **5** videos are ignored.

## 3. Folder structure
No fixed structure is required.
Put the dataset videos anywhere under `videos/` (nested folders are OK) — the pipeline scans this directory recursively.
