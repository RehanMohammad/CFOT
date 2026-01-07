import torch
import GPUtil
from typing import Dict, Any

class GPUMonitor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get comprehensive GPU statistics"""
        stats = {}
        
        if self.device.type == 'cuda':
            # Current GPU usage
            gpus = GPUtil.getGPUs()
            current_gpu = gpus[0]  # Assuming single GPU or primary GPU
            
            # PyTorch memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            stats = {
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_utilization': current_gpu.load * 100,  # %
                'gpu_memory_used': current_gpu.memoryUsed,  # MB
                'gpu_memory_total': current_gpu.memoryTotal,  # MB
                'gpu_memory_utilization': current_gpu.memoryUtil * 100,  # %
                'pytorch_allocated_gb': allocated,
                'pytorch_reserved_gb': reserved,
                'pytorch_max_allocated_gb': max_allocated,
                'cuda_version': torch.version.cuda
            }
        
        return stats
    
    def print_gpu_stats(self, prefix=""):
        """Print formatted GPU statistics"""
        stats = self.get_gpu_stats()
        if stats:
            print(f"{prefix} GPU Stats: "
                  f"Util: {stats['gpu_utilization']:.1f}% | "
                  f"Mem: {stats['gpu_memory_used']}/{stats['gpu_memory_total']} MB "
                  f"({stats['gpu_memory_utilization']:.1f}%) | "
                  f"PyTorch: {stats['pytorch_allocated_gb']:.2f} GB allocated")