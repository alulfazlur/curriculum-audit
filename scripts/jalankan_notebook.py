"""Jalankan notebook dari awal sampai akhir tanpa VS Code, lalu simpan output-nya ke file notebook.

    .venv/bin/python scripts/jalankan_notebook.py Audit_Kurikulum_EN_CA_Pipeline_ID.ipynb

Berhenti pada cell pertama yang error; output cell yang sudah jalan tetap disimpan.
"""
import json, os, sys, time

from jupyter_client.manager import start_new_kernel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8"))
km, kc = start_new_kernel(kernel_name="python3", cwd=ROOT)
count, failed = 0, False
try:
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        count += 1
        t0 = time.time()
        msg_id = kc.execute("".join(cell["source"]))
        outputs = []
        while True:
            msg = kc.get_iopub_msg(timeout=3600)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "status" and c["execution_state"] == "idle":
                break
            if t == "stream":
                if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == c["name"]:
                    outputs[-1]["text"] += c["text"]
                else:
                    outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
            elif t == "execute_result":
                outputs.append({"output_type": "execute_result", "execution_count": count,
                                "data": c["data"], "metadata": {}})
            elif t == "display_data":
                outputs.append({"output_type": "display_data", "data": c["data"], "metadata": {}})
            elif t == "error":
                outputs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"],
                                "traceback": c["traceback"]})
                failed = True
        for o in outputs:
            if o["output_type"] == "stream":
                o["text"] = o["text"].splitlines(keepends=True)
        cell["outputs"], cell["execution_count"] = outputs, count
        print(f"cell {i:2d} [{count:2d}] {time.time() - t0:7.1f}s {'ERROR' if failed else 'ok'}", flush=True)
        if failed:
            err = outputs[-1]
            print(err["ename"], err["evalue"], flush=True)
            break
finally:
    km.shutdown_kernel(now=True)
    tmp = path + ".tmp"
    json.dump(nb, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)
sys.exit(1 if failed else 0)
