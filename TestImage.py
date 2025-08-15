import os

test_dir = "data/Testing"
classes = ["Green_Curry", "Khao_phat", "Khao_Soi", "Massaman_Curry",
           "Pad_Krapraw", "Pad_Thai", "SomTum", "Tom_yum"]

for cls in classes:
    path = os.path.join(test_dir, cls)
    print(f"{cls}: {len(os.listdir(path))} files")