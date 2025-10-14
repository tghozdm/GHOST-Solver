# =============================================================================
# Hukuki Başlık (AGPLv3) - Zorunlu Kısım
# =============================================================================

# Copyright (C) 2025 [Adınız Soyadınız]

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# =============================================================================
# Modül Açıklaması (Mühendislik İçin)
# =============================================================================

"""
GHOST-Solver Numba CUDA Hyper-Operator Kernel Core Module.

Contains the highly-optimized JIT-compiled CUDA kernel for the Jacobi-based 
iterative linear system solving method (Ax=b).

Licensed under the AGPLv3.
"""

    """
╔════════════════════════════════════════════════════════════════════╗
║                     GHOST-SOLVER v2.0                             ║
║      GPU-Heavy Operations for Solving Triangular Systems          ║
║                   Jacobi Method Implementation                     ║
║                                                                    ║
║  Formül: x_i^(k+1) = (1/A_ii)(b_i - Σ_{j≠i} A_ij·x_j^(k))       ║
║                                                                    ║
║  Autor: CUDA Jacobi Team                                          ║
║  Tarih: 2024                                                       ║
║  Durum: Production Ready ✓                                        ║
╚════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from numba import cuda
import time
from typing import Tuple, Optional


# =============================================================================
# JACOBI KERNEL - GPU ÜZERİNDE PARALEL HESAPLAMA
# =============================================================================

@cuda.jit
def jacobi_kernel(X_new, X_old, A, B, N):
    """
    Jacobi iterasyon GPU çekirdeği.
    
    Her GPU thread bir denklem satırını paralel olarak çözer.
    
    Parametreler:
    -----------
    X_new : ndarray (N, 1)
        Yeni x vektörü (çıktı)
    X_old : ndarray (N, 1)
        Eski x vektörü (girdi) - bir önceki iterasyondan
    A : ndarray (N, N)
        Katsayı matrisi
    B : ndarray (N,)
        Sağ taraf vektörü
    N : int
        Sistem boyutu
        
    Formül:
    ------
    x_i^(k+1) = (1/A_ii) * (b_i - Σ_{j≠i} A_ij * x_j^(k))
    
    Nota: 
    - X_old sadece okunur (ESKİ iterasyondan)
    - X_new yazılır (YENİ iterasyon için)
    - Her thread idx için: 0 <= idx < N
    """
    
    # 1D grid indexi: her thread bir satır işler
    idx = cuda.grid(1)
    
    # Thread kontrol: eğer idx < N ise çalış
    if idx < N:
        # Σ A_ij * x_j^(k) toplamını hesapla (j ≠ i)
        sum_val = 0.0
        
        for j in range(N):
            if j != idx:  # j ≠ i
                # Tüm terimler ESKİ x değerleri ile hesaplanıyor
                sum_val += A[idx, j] * X_old[j, 0]
        
        # Köşegen eleman A_ii
        diag = A[idx, idx]
        
        # Singular matrix kontrolü
        if abs(diag) < 1e-14:
            raise ValueError(f"Singular matrix at row {idx}")
        
        # Ana formül: x_i^(k+1) = (b_i - Σ) / A_ii
        X_new[idx, 0] = (B[idx] - sum_val) / diag


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def compute_residual(A: np.ndarray, X: np.ndarray, B: np.ndarray) -> float:
    """
    Residual hesapla: ||Ax - B||_2
    
    Parametreler:
    -----------
    A : ndarray (N, N)
        Katsayı matrisi
    X : ndarray (N, 1)
        Çözüm vektörü
    B : ndarray (N,)
        Sağ taraf
        
    Dönüş:
    -----
    float
        L2 norm residual
    """
    return np.linalg.norm(A @ X - B)


def check_diagonal_dominance(A: np.ndarray) -> Tuple[bool, float]:
    """
    Diagonal-dominant kontrol.
    
    Koşul: |A_ii| > Σ_{j≠i} |A_ij| (her i için)
    
    Parametreler:
    -----------
    A : ndarray (N, N)
        Katsayı matrisi
        
    Dönüş:
    -----
    Tuple[bool, float]
        (is_dd, ratio) - DD ise True, max ratio
    """
    N = A.shape[0]
    ratios = []
    
    for i in range(N):
        diag = abs(A[i, i])
        off_diag_sum = sum(abs(A[i, j]) for j in range(N) if j != i)
        
        if off_diag_sum > 0:
            ratio = diag / off_diag_sum
            ratios.append(ratio)
    
    min_ratio = min(ratios) if ratios else 0
    is_dd = min_ratio > 1.0
    
    return is_dd, min_ratio


# =============================================================================
# ANA ÇÖZÜCÜ FONKSİYONU
# =============================================================================

def solve_jacobi_gpu(
    A: np.ndarray,
    B: np.ndarray,
    X_initial: Optional[np.ndarray] = None,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    threads_per_block: int = 256,
    verbose: bool = True
) -> Tuple[np.ndarray, int, float, dict]:
    """
    GHOST-Solver: GPU üzerinde Jacobi yöntemi ile lineer sistemi çöz.
    
    Ax = b sistemini çöz iteratif Jacobi yöntemi kullanarak.
    GPU parallelizasyonu ile hızlanma sağlar.
    
    Parametreler:
    -----------
    A : ndarray (N, N)
        Katsayı matrisi (diagonal-dominant olmalı)
    B : ndarray (N,) veya (N, 1)
        Sağ taraf vektörü
    X_initial : ndarray (N, 1), optional
        İlk tahmin (default: sıfır vektörü)
    max_iterations : int
        Maksimum iterasyon sayısı (default: 500)
    tolerance : float
        Yakınsama toleransı (default: 1e-8)
    threads_per_block : int
        GPU threads/block (default: 256)
    verbose : bool
        Detaylı çıktı göster (default: True)
        
    Dönüş:
    -----
    Tuple[ndarray, int, float, dict]
        - X : çözüm vektörü (N, 1)
        - iterations : kullanılan iterasyon sayısı
        - final_residual : son residual
        - info : ek bilgiler (istatistikler)
        
    Notlar:
    ------
    - A'nın diagonal-dominant olması yakınsama için gereklidir
    - GPU'da transfer overhead küçük N'de önemli olabilir (N>100 önerilir)
    - CUDA compute capability 3.0+ gereklidir
    
    Örnek:
    -----
    >>> A = np.array([[10, -1, 2], [-1, 11, -1], [2, -1, 10]], dtype=np.float64)
    >>> B = np.array([6, 25, -11], dtype=np.float64)
    >>> X, iters, res, info = solve_jacobi_gpu(A, B)
    >>> print(f"Çözüm: {X.ravel()}")
    >>> print(f"İterasyonlar: {iters}")
    """
    
    # ─────────────────────────────────────────────────────────────────
    # KONTROL VE HAZIRLIK
    # ─────────────────────────────────────────────────────────────────
    
    N = A.shape[0]
    
    # B shape kontrolü (1D → 2D)
    if B.ndim == 2:
        B = B.ravel()
    
    # İlk tahmin
    if X_initial is None:
        X_initial = np.zeros((N, 1), dtype=np.float64)
    
    # Relative tolerance (mutlak'a çevir)
    B_norm = np.linalg.norm(B)
    tol_abs = tolerance * B_norm if B_norm > 0 else tolerance
    
    # İstatistikler
    info = {
        'system_size': N,
        'b_norm': B_norm,
        'tolerance': tolerance,
        'tol_abs': tol_abs,
        'blocks': (N + threads_per_block - 1) // threads_per_block,
        'threads_per_block': threads_per_block,
        'residuals': [],
        'gpu_memory_used': 0,
        'total_time': 0
    }
    
    if verbose:
        print(f"\n{'='*75}")
        print(f"{'GHOST-SOLVER: Jacobi Yöntemi (GPU)':<50}")
        print(f"{'='*75}")
        print(f"Sistem boyutu: {N}×{N}")
        print(f"Tolerans: {tolerance:.2e}")
        print(f"Max iterasyon: {max_iterations}")
        print(f"GPU Config: {info['blocks']} blocks × {threads_per_block} threads")
        
        # Diagonal-dominant kontrol
        is_dd, dd_ratio = check_diagonal_dominance(A)
        print(f"Diagonal-dominant: {'✓ Evet' if is_dd else '✗ Hayır'} (ratio: {dd_ratio:.4f})")
        print(f"{'─'*75}")
    
    # ─────────────────────────────────────────────────────────────────
    # GPU'YA TRANSFER
    # ─────────────────────────────────────────────────────────────────
    
    start_total = time.time()
    
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    X_old_gpu = cuda.to_device(X_initial.copy())
    X_new_gpu = cuda.to_device(np.zeros_like(X_initial))
    
    # GPU hafıza tahmini
    info['gpu_memory_used'] = (A.nbytes + B.nbytes + 
                               X_initial.nbytes * 2) / (1024**2)  # MB
    
    # CUDA konfigürasyonu
    blocks_per_grid = info['blocks']
    threads_per_block = threads_per_block
    
    if verbose:
        print(f"GPU Hafıza: ~{info['gpu_memory_used']:.2f} MB")
        print(f"\n{'İter':<8} {'Residual':<18} {'Değişim':<18} {'Durum':<15}")
        print(f"{'─'*75}")
    
    # ─────────────────────────────────────────────────────────────────
    # İTERASYONLAR
    # ─────────────────────────────────────────────────────────────────
    
    X_prev = None
    final_residual = float('inf')
    converged_iter = -1
    
    for k in range(max_iterations):
        # GPU kernel çalıştır
        jacobi_kernel[blocks_per_grid, threads_per_block](
            X_new_gpu, X_old_gpu, A_gpu, B_gpu, N
        )
        
        # ✅ KRİTİK: SWAP (X_old ← X_new)
        X_old_gpu, X_new_gpu = X_new_gpu, X_old_gpu
        
        # Yakınsama kontrolü (seyrek - overhead azaltmak için)
        if k % 20 == 0 or k < 5:
            # GPU'dan CPU'ya kopyala
            X_result = X_old_gpu.copy_to_host()
            residual = compute_residual(A, X_result, B)
            info['residuals'].append(residual)
            
            # Değişim hesapla
            if X_prev is not None:
                change = np.linalg.norm(X_result - X_prev)
            else:
                change = np.linalg.norm(X_result)
            
            X_prev = X_result.copy()
            
            # Durum belirleme
            if residual < tol_abs:
                status = "CONVERGED ✓"
                converged_iter = k
                final_residual = residual
            else:
                status = "Iterating"
            
            if verbose:
                print(f"{k:<8} {residual:<18.8e} {change:<18.8e} {status:<15}")
            
            # Yakınsama sağlandı mı?
            if residual < tol_abs:
                if verbose:
                    print(f"{'─'*75}")
                    print(f"✓ Yakınsama sağlandı {k} iterasyonda!")
                
                info['converged'] = True
                info['converged_iter'] = k
                
                return X_result, k, residual, info
    
    # ─────────────────────────────────────────────────────────────────
    # MAKSIMUM İTERASYONA ULAŞTI
    # ─────────────────────────────────────────────────────────────────
    
    X_result = X_old_gpu.copy_to_host()
    final_residual = compute_residual(A, X_result, B)
    
    if verbose:
        print(f"{'─'*75}")
        print(f"✗ Maksimum iterasyona ulaşıldı ({max_iterations})")
        print(f"  Son residual: {final_residual:.8e}")
    
    info['converged'] = False
    info['converged_iter'] = -1
    info['total_time'] = time.time() - start_total
    
    return X_result, max_iterations, final_residual, info


# =============================================================================
# TEST FONKSIYONLARI
# =============================================================================

def test_basic_system():
    """Test 1: Basit 4×4 Sistem"""
    print("\n" + "▶"*50)
    print("TEST 1: 4×4 Diagonal-Dominant Sistem")
    print("▶"*50)
    
    A = np.array([
        [10.0, -1.0, 2.0, 0.0],
        [-1.0, 11.0, -1.0, 3.0],
        [2.0, -1.0, 10.0, -1.0],
        [0.0, 3.0, -1.0, 8.0]
    ], dtype=np.float64)
    
    B = np.array([6.0, 25.0, -11.0, 15.0], dtype=np.float64)
    X_init = np.zeros((4, 1), dtype=np.float64)
    
    # GHOST çözüm
    X_ghost, iters, res, info = solve_jacobi_gpu(
        A, B, X_init, 
        max_iterations=500, 
        tolerance=1e-8,
        verbose=True
    )
    
    # NumPy referans
    from scipy.linalg import solve as sp_solve
    X_numpy = sp_solve(A, B)
    
    # Karşılaştırma
    error = np.linalg.norm(X_ghost - X_numpy.reshape(-1, 1))
    
    print(f"\nSonuç:")
    print(f"  GHOST:  {X_ghost.ravel()}")
    print(f"  NumPy:  {X_numpy}")
    print(f"  Error:  {error:.8e}")
    print(f"  ✓ BAŞARILI" if error < 1e-6 else f"  ✗ BAŞARISIZ")
    
    return error < 1e-6


def test_medium_system():
    """Test 2: Orta Sistem (200×200)"""
    print("\n" + "▶"*50)
    print("TEST 2: 200×200 Diagonal-Dominant Sistem")
    print("▶"*50)
    
    np.random.seed(42)
    N = 200
    
    A = np.random.randn(N, N) * 0.1
    for i in range(N):
        A[i, i] = N * 1.5
    
    B = np.ones(N)
    X_init = np.zeros((N, 1), dtype=np.float64)
    
    # GHOST çözüm
    X_ghost, iters, res, info = solve_jacobi_gpu(
        A, B, X_init,
        max_iterations=500,
        verbose=False
    )
    
    # NumPy referans
    from scipy.linalg import solve as sp_solve
    X_numpy = sp_solve(A, B)
    
    error = np.linalg.norm(X_ghost - X_numpy.reshape(-1, 1))
    
    print(f"\nSonuç:")
    print(f"  İterasyonlar: {iters}")
    print(f"  Residual: {res:.8e}")
    print(f"  Error: {error:.8e}")
    print(f"  GPU Hafıza: {info['gpu_memory_used']:.2f} MB")
    print(f"  ✓ BAŞARILI" if error < 1e-5 else f"  ✗ BAŞARISIZ")
    
    return error < 1e-5


def test_large_system():
    """Test 3: Büyük Sistem (500×500)"""
    print("\n" + "▶"*50)
    print("TEST 3: 500×500 Diagonal-Dominant Sistem")
    print("▶"*50)
    
    np.random.seed(42)
    N = 500
    
    A = np.random.randn(N, N) * 0.1
    for i in range(N):
        A[i, i] = N * 1.5
    
    B = np.ones(N)
    X_init = np.zeros((N, 1), dtype=np.float64)
    
    # GHOST çözüm
    X_ghost, iters, res, info = solve_jacobi_gpu(
        A, B, X_init,
        max_iterations=500,
        verbose=False
    )
    
    # NumPy referans
    from scipy.linalg import solve as sp_solve
    X_numpy = sp_solve(A, B)
    
    error = np.linalg.norm(X_ghost - X_numpy.reshape(-1, 1))
    
    print(f"\nSonuç:")
    print(f"  İterasyonlar: {iters}")
    print(f"  Residual: {res:.8e}")
    print(f"  Error: {error:.8e}")
    print(f"  ✓ BAŞARILI" if error < 1e-5 else f"  ✗ BAŞARISIZ")
    
    return error < 1e-5


# =============================================================================
# ANA PROGRAM
# =============================================================================

if __name__ == '__main__':
    print("\n")
    print("╔" + "="*73 + "╗")
    print("║" + " "*15 + "GHOST-SOLVER v2.0 - Jacobi CUDA Yöntemi" + " "*20 + "║")
    print("║" + " "*20 + "Üretim Hazır Kod (Production Ready)" + " "*19 + "║")
    print("╚" + "="*73 + "╝")
    
    try:
        # Testleri çalıştır
        test1 = test_basic_system()
        test2 = test_medium_system()
        test3 = test_large_system()
        
        # Özet
        print("\n" + "="*75)
        print("TEST ÖZETİ")
        print("="*75)
        print(f"Test 1 (4×4):       {'✓ BAŞARILI' if test1 else '✗ BAŞARISIZ'}")
        print(f"Test 2 (200×200):   {'✓ BAŞARILI' if test2 else '✗ BAŞARISIZ'}")
        print(f"Test 3 (500×500):   {'✓ BAŞARILI' if test3 else '✗ BAŞARISIZ'}")
        
        if test1 and test2 and test3:
            print(f"\n{'='*75}")
            print(f"✓✓✓ TÜM TESTLER BAŞARILI - GHOST-SOLVER HAZIR ✓✓✓")
            print(f"{'='*75}\n")
        
    except Exception as e:
        print(f"\n✗ HATA: {e}")
        import traceback
        traceback.print_exc()
