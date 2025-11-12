import sys
import os
import cbor2

SCALE = 9.81

def main(infile):
    with open(infile, "rb") as f:
        data = cbor2.load(f)

    payload = data.get("payload", data)
    sensors = payload["sensors"]
    values = payload["values"]

    name_to_idx = {s["name"]: i for i, s in enumerate(sensors)}
    accx_i = name_to_idx["accX"]
    accy_i = name_to_idx["accY"]
    accz_i = name_to_idx["accZ"]

    new_values = []
    for row in values:
        accx = row[accx_i] / SCALE
        accy = row[accy_i] / SCALE
        accz = row[accz_i] / SCALE

        catX = accy
        catY = accz
        catZ = -accx

        new_values.append([catX, catY, catZ])

    payload["sensors"] = [
        {"name": "accX", "units": "g"},
        {"name": "accY", "units": "g"},
        {"name": "accZ", "units": "g"},
    ]
    payload["values"] = new_values

    outname = os.path.splitext(infile)[0] + "_acc.cbor"
    with open(outname, "wb") as f:
        cbor2.dump(data, f)

    print("written:", outname)

if __name__ == "__main__":
    main(sys.argv[1])
