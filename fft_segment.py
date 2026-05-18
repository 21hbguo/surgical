import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img_path = '/home/guo/project/ssl4mis/data/endovis2017/data/images/seq_1_frame000.png'
    output_path = '/home/guo/project/ssl4mis/data/output/fft_segment.png'

    img = Image.open(img_path).convert('RGB').resize((224, 224), Image.BILINEAR)
    img_np = np.array(img)

    fft = np.fft.fft2(img_np, axes=(0, 1))
    fft_shift = np.fft.fftshift(fft)

    h, w = img_np.shape[:2]
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cy**2 + cx**2)

    phase = np.angle(fft_shift)

    # 分段：0-3%, 3-10%, 10-20%, ..., 90-100%
    bands = [('Original', np.ones_like(dist, dtype=bool))]
    # 0-3%
    bands.append(('Freq (0-3%)', dist < max_dist * 0.03))
    # 3-10%
    bands.append(('Freq (3-10%)', (dist >= max_dist * 0.03) & (dist < max_dist * 0.10)))
    # 10-20%, ..., 90-100%
    for i in range(1, 10):
        low = i * 0.1
        high = (i + 1) * 0.1
        mask = (dist >= max_dist * low) & (dist < max_dist * high)
        bands.append((f'Freq ({int(low*100)}-{int(high*100)}%)', mask))

    # 累积：0-3%, 0-10%, 0-20%, ..., 0-100%
    cum_bands = [
        ('Cumulative (0-3%)', dist < max_dist * 0.03),
        ('Cumulative (0-10%)', dist < max_dist * 0.10),
    ]
    for i in range(1, 10):
        high = (i + 1) * 0.1
        cum_bands.append((f'Cumulative (0-{int(high*100)}%)', dist < max_dist * high))

    # 反向累积：100-3%, 100-10%, 100-20%, ..., 0-100%
    rev_cum_bands = [
        ('Reverse (3-100%)', dist >= max_dist * 0.03),
        ('Reverse (10-100%)', dist >= max_dist * 0.10),
    ]
    for i in range(1, 10):
        low = i * 0.1
        rev_cum_bands.append((f'Reverse ({int(low*100)}-100%)', dist >= max_dist * low))

    fig, axes = plt.subplots(4, 12, figsize=(36, 16))

    # 第1、2行：频谱 + 分段重建
    for i, (title, mask) in enumerate(bands):
        mask_3d = mask[:, :, np.newaxis]

        # 频谱
        spectrum = np.abs(fft_shift).mean(axis=2)
        if title == 'Original':
            spectrum_display = np.log1p(spectrum)
            spectrum_display = (spectrum_display / spectrum_display.max() * 255)
        else:
            spectrum_masked = spectrum * mask
            spectrum_max = spectrum_masked.max()
            if spectrum_max > 0:
                spectrum_display = (spectrum_masked / spectrum_max * 255)
            else:
                spectrum_display = spectrum_masked
        spectrum_display = np.clip(spectrum_display, 0, 255).astype(np.uint8)
        axes[0, i].imshow(spectrum_display, cmap='gray')
        axes[0, i].set_title(title, fontsize=7)
        axes[0, i].axis('off')

        # 重建
        if title == 'Original':
            result = img_np
        else:
            amplitude = np.abs(fft_shift) * mask_3d
            complex_img = amplitude * np.exp(1j * phase)
            complex_img = np.fft.ifftshift(complex_img)
            img_recon = np.fft.ifft2(complex_img, axes=(0, 1))
            img_recon = np.abs(img_recon)
            for c in range(3):
                ch_max = img_recon[:, :, c].max()
                if ch_max > 0:
                    img_recon[:, :, c] = img_recon[:, :, c] / ch_max * 255
            result = np.clip(img_recon, 0, 255).astype(np.uint8)

        axes[1, i].imshow(result)
        axes[1, i].set_title(title, fontsize=7)
        axes[1, i].axis('off')

    # 第3行：正向累积
    for i, (title, mask) in enumerate(cum_bands):
        mask_3d = mask[:, :, np.newaxis]
        amplitude = np.abs(fft_shift) * mask_3d
        complex_img = amplitude * np.exp(1j * phase)
        complex_img = np.fft.ifftshift(complex_img)
        img_recon = np.fft.ifft2(complex_img, axes=(0, 1))
        img_recon = np.abs(img_recon)
        for c in range(3):
            ch_max = img_recon[:, :, c].max()
            if ch_max > 0:
                img_recon[:, :, c] = img_recon[:, :, c] / ch_max * 255
        result = np.clip(img_recon, 0, 255).astype(np.uint8)

        axes[2, i + 1].imshow(result)
        axes[2, i + 1].set_title(title, fontsize=7)
        axes[2, i + 1].axis('off')

    axes[2, 0].axis('off')

    # 第4行：反向累积
    for i, (title, mask) in enumerate(rev_cum_bands):
        mask_3d = mask[:, :, np.newaxis]
        amplitude = np.abs(fft_shift) * mask_3d
        complex_img = amplitude * np.exp(1j * phase)
        complex_img = np.fft.ifftshift(complex_img)
        img_recon = np.fft.ifft2(complex_img, axes=(0, 1))
        img_recon = np.abs(img_recon)
        for c in range(3):
            ch_max = img_recon[:, :, c].max()
            if ch_max > 0:
                img_recon[:, :, c] = img_recon[:, :, c] / ch_max * 255
        result = np.clip(img_recon, 0, 255).astype(np.uint8)

        axes[3, i + 1].imshow(result)
        axes[3, i + 1].set_title(title, fontsize=7)
        axes[3, i + 1].axis('off')

    axes[3, 0].axis('off')

    axes[0, 0].set_ylabel('Spectrum', fontsize=11)
    axes[1, 0].set_ylabel('Band\nReconstruction', fontsize=11)
    axes[2, 0].set_ylabel('Cumulative\n(0-X%)', fontsize=11)
    axes[3, 0].set_ylabel('Reverse\n(X-100%)', fontsize=11)

    plt.suptitle('FFT Band Segmentation (0-3%, 3-10%, 10-20%, ..., 90-100%)', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')

if __name__ == '__main__':
    main()
