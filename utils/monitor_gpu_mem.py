import jax

device = jax.devices()[0]  # 查看第一个 GPU
stats = device.memory_stats()
print(f"当前实际使用: {stats['bytes_in_use'] / 1024**2:.2f} MB")
print(f"历史峰值使用: {stats['peak_bytes_in_use'] / 1024**2:.2f} MB")
print(f"XLA 已预分配: {stats['bytes_reserved'] / 1024**2:.2f} MB")