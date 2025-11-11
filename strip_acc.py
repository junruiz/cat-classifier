import cbor2
import sys
from pathlib import Path

infile = Path(sys.argv[1])
outfile = infile.with_name(infile.stem + "_acc.cbor")

with infile.open("rb") as f:
    data = cbor2.load(f)

payload = data.get("payload", data)

sensors = payload.get("sensors")
values = payload.get("values") or payload.get("data")

if sensors is None or values is None:
    raise RuntimeError("unrecognized CBOR structure")

keep_idx = []
for i, s in enumerate(sensors):
    name = s.get("name", "")
    if name in ("accX", "accY", "accZ"):
        keep_idx.append(i)

payload["sensors"] = [sensors[i] for i in keep_idx]

new_values = []
for row in values:
    new_values.append([row[i] for i in keep_idx])

payload["values"] = new_values

with outfile.open("wb") as f:
    cbor2.dump(data, f)

print("written:", outfile)
