# train3.ps1 — 3-way sharded LoRA training, one shard pinned per GPU
$env:TMP = "D:\doomsrv"; $env:TEMP = "D:\doomsrv"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$procs = @()
foreach ($i in 0..2) {
  $p = Start-Process -FilePath "cmd.exe" -ArgumentList @(
    "/c", "set CUDA_VISIBLE_DEVICES=$i&& set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True&& D:\doom-venv\Scripts\python.exe train_lora.py --shard $i --nshards 3 --out D:\doomsrv\spine-lora-s$i > D:\doomsrv\s$i.log 2> D:\doomsrv\s$i.err"
  ) -WorkingDirectory "D:\doomsrv" -WindowStyle Hidden -PassThru
  $procs += $p
  Write-Output "shard $i -> PID $($p.Id) on GPU $i"
}
$procs | Wait-Process
Write-Output "all shards done"
cmd /c "cd /d D:\doomsrv&& set CUDA_VISIBLE_DEVICES=2&& D:\doom-venv\Scripts\python.exe merge_loras.py > D:\doomsrv\merge.log 2>&1"
Write-Output "merge done"
