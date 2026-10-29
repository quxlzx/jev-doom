@echo off
cd /d D:\doomsrv
llama-server.exe -m D:\doomsrv\Qwen3-8B-Q4_K_M.gguf -ngl 99 -dev CUDA2 -c 4096 -t 8 --host 127.0.0.1 --port 8182 > brain.log 2> brain.err
