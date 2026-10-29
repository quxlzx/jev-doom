@echo off
cd /d D:\doomsrv
llama-server.exe -m D:\doomsrv\Qwen3-1.7B-Q8_0.gguf --lora D:\doomsrv\spine-lora.gguf -ngl 99 -dev CUDA1 -c 2048 -t 8 --host 127.0.0.1 --port 8181 --grammar-file D:\doomsrv\reflex_pick.gbnf > spine.log 2> spine.err
