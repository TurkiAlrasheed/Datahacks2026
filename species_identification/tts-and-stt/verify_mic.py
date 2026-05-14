import sounddevice as sd
print(sd.query_devices())
print("default input:", sd.default.device[0])