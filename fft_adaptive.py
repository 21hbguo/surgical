import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img_path = '/home/guo/project/ssl4mis/data/endovis2017/data/images/seq_1_frame000.png'
    output_path = '/home/guo/project/ssl4mis/data/output/fft_adaptive.png'

    img = Image.open(img_path).convert('RGB').resize((224, 224), Image.BILINEAR)
    img_np = np.array(img)

    fft = np.fft.fft2(img_np, axes=(0, 1))
    fft_shift = np.fft.fftshift(fft)

    h, w = img_np.shape[:2]
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cy**2 + cx**2)

    # 计算径向能量分布
    spectrum = np.abs(fft_shift).mean(axis=2)
    max_radius = int(max_dist)

    # 每个半径的平均能量
    radial_energy = np.zeros(max_radius)
    for r in range(max_radius):
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        if mask.any():
            radial_energy[r] = spectrum[mask].mean()

    # 方法：累积能量法
    # 计算每个半径处的累积能量占比
    total_energy = radial_energy.sum()
    cumulative_energy = np.cumsum(radial_energy) / total_energy

    # r_min: 累积能量达到5%的位置（跳过中心直流）
    r_min = 1
    for i in range(1, len(cumulative_energy)):
        if cumulative_energy[i] >= 0.05:
            r_min = i
            break

    # r_max: 累积能量达到85%的位置（保留主要能量）
    r_max = len(radial_energy) - 1
    for i in range(1, len(cumulative_energy)):
        if cumulative_energy[i] >= 0.85:
            r_max = i
            break

    # 确保范围合理
    r_min = max(3, r_min)
    r_max = min(len(radial_energy) - 1, r_max)

    # 如果范围太窄，至少保证有20个像素宽度
    if r_max - r_min < 20:
        r_max = min(len(radial_energy) - 1, r_min + 20)

    # 计算梯度用于显示
    gradient = np.diff(radial_energy)

    print(f"Energy drop at radius: {r_min}")
    print(f"Energy stable at radius: {r_max}")
    print(f"Optimal band: {r_min/max_dist*100:.1f}% - {r_max/max_dist*100:.1f}%")

    # 创建图
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # 第1行：频谱和径向能量
    # 原图频谱
    spectrum_display = np.log1p(spectrum)
    spectrum_display = (spectrum_display / spectrum_display.max() * 255).astype(np.uint8)
    axes[0, 0].imshow(spectrum_display, cmap='gray')
    axes[0, 0].set_title('Original Spectrum')
    axes[0, 0].axis('off')

    # 标记r_min和r_max的频谱
    spectrum_annotated = spectrum_display.copy()
    # 画圆圈标记r_min和r_max
    theta = np.linspace(0, 2*np.pi, 100)
    for ax_idx, (r, color, label) in enumerate([(r_min, 'green', f'r_min={r_min}'), (r_max, 'red', f'r_max={r_max}')]):
        circle_y = cy + r * np.sin(theta)
        circle_x = cx + r * np.cos(theta)
        # 在频谱图上标记
        axes[0, 0].plot(circle_x, circle_y, color=color, linewidth=1.5, label=label)

    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].set_title('Optimal Band Marked')

    # 径向能量分布
    axes[0, 1].plot(radial_energy, 'b-', linewidth=2)
    axes[0, 1].axvline(x=r_min, color='green', linestyle='--', label=f'r_min={r_min} ({r_min/max_dist*100:.1f}%)')
    axes[0, 1].axvline(x=r_max, color='red', linestyle='--', label=f'r_max={r_max} ({r_max/max_dist*100:.1f}%)')
    axes[0, 1].set_xlabel('Radius')
    axes[0, 1].set_ylabel('Energy')
    axes[0, 1].set_title('Radial Energy Distribution')
    axes[0, 1].legend()

    # 梯度
    axes[0, 2].plot(gradient, 'g-', linewidth=2)
    axes[0, 2].axvline(x=r_min, color='green', linestyle='--', label='r_min (max drop)')
    axes[0, 2].axvline(x=r_max, color='red', linestyle='--', label='r_max (stable)')
    axes[0, 2].set_xlabel('Radius')
    axes[0, 2].set_ylabel('Gradient')
    axes[0, 2].set_title('Energy Gradient (1st Derivative)')
    axes[0, 2].legend()

    # 带通掩码
    mask_bandpass = (dist >= r_min) & (dist < r_max)
    mask_display = (mask_bandpass * 255).astype(np.uint8)
    axes[0, 3].imshow(mask_display, cmap='gray')
    axes[0, 3].set_title(f'Bandpass Mask\n({r_min/max_dist*100:.1f}%-{r_max/max_dist*100:.1f}%)')
    axes[0, 3].axis('off')

    # 带通频谱
    spectrum_bp = spectrum * mask_bandpass
    spectrum_bp_display = np.log1p(spectrum_bp)
    if spectrum_bp_display.max() > 0:
        spectrum_bp_display = (spectrum_bp_display / spectrum_bp_display.max() * 255)
    axes[0, 4].imshow(spectrum_bp_display.astype(np.uint8), cmap='gray')
    axes[0, 4].set_title('Bandpass Spectrum')
    axes[0, 4].axis('off')

    # 第2行：重建对比
    phase = np.angle(fft_shift)

    # 原图
    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')

    # 低频保留（0-r_min）
    mask_low = dist < r_min
    mask_3d = mask_low[:, :, np.newaxis]
    amplitude = np.abs(fft_shift) * mask_3d
    complex_img = amplitude * np.exp(1j * phase)
    complex_img = np.fft.ifftshift(complex_img)
    img_low = np.fft.ifft2(complex_img, axes=(0, 1))
    img_low = np.abs(img_low)
    for c in range(3):
        ch_max = img_low[:, :, c].max()
        if ch_max > 0:
            img_low[:, :, c] = img_low[:, :, c] / ch_max * 255
    axes[1, 1].imshow(np.clip(img_low, 0, 255).astype(np.uint8))
    axes[1, 1].set_title(f'Low Freq Only\n(0-{r_min/max_dist*100:.1f}%)')
    axes[1, 1].axis('off')

    # 带通保留（r_min-r_max）
    mask_3d = mask_bandpass[:, :, np.newaxis]
    amplitude = np.abs(fft_shift) * mask_3d
    complex_img = amplitude * np.exp(1j * phase)
    complex_img = np.fft.ifftshift(complex_img)
    img_bp = np.fft.ifft2(complex_img, axes=(0, 1))
    img_bp = np.abs(img_bp)
    for c in range(3):
        ch_max = img_bp[:, :, c].max()
        if ch_max > 0:
            img_bp[:, :, c] = img_bp[:, :, c] / ch_max * 255
    axes[1, 2].imshow(np.clip(img_bp, 0, 255).astype(np.uint8))
    axes[1, 2].set_title(f'Bandpass Only\n({r_min/max_dist*100:.1f}%-{r_max/max_dist*100:.1f}%)')
    axes[1, 2].axis('off')

    # 高频保留（r_max-100%）
    mask_high = dist >= r_max
    mask_3d = mask_high[:, :, np.newaxis]
    amplitude = np.abs(fft_shift) * mask_3d
    complex_img = amplitude * np.exp(1j * phase)
    complex_img = np.fft.ifftshift(complex_img)
    img_high = np.fft.ifft2(complex_img, axes=(0, 1))
    img_high = np.abs(img_high)
    for c in range(3):
        ch_max = img_high[:, :, c].max()
        if ch_max > 0:
            img_high[:, :, c] = img_high[:, :, c] / ch_max * 255
    axes[1, 3].imshow(np.clip(img_high, 0, 255).astype(np.uint8))
    axes[1, 3].set_title(f'High Freq Only\n({r_max/max_dist*100:.1f}%-100%)')
    axes[1, 3].axis('off')

    # 带通去除（低频+高频）
    mask_not_bp = ~mask_bandpass
    mask_3d = mask_not_bp[:, :, np.newaxis]
    amplitude = np.abs(fft_shift) * mask_3d
    complex_img = amplitude * np.exp(1j * phase)
    complex_img = np.fft.ifftshift(complex_img)
    img_not_bp = np.fft.ifft2(complex_img, axes=(0, 1))
    img_not_bp = np.abs(img_not_bp)
    for c in range(3):
        ch_max = img_not_bp[:, :, c].max()
        if ch_max > 0:
            img_not_bp[:, :, c] = img_not_bp[:, :, c] / ch_max * 255
    axes[1, 4].imshow(np.clip(img_not_bp, 0, 255).astype(np.uint8))
    axes[1, 4].set_title(f'Bandpass Removed\n(Not {r_min/max_dist*100:.1f}%-{r_max/max_dist*100:.1f}%)')
    axes[1, 4].axis('off')

    plt.suptitle(f'Adaptive Bandpass Filter: r_min={r_min} ({r_min/max_dist*100:.1f}%), r_max={r_max} ({r_max/max_dist*100:.1f}%)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')

if __name__ == '__main__':
    main()
