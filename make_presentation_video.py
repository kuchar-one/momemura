import os
import glob
import subprocess
import imageio.v2 as imageio  # fallback if needed, but we use typical imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_MP4 = "presentation_video.mp4"
TOTAL_FRAMES = 1440
FPS = 24

TOTAL_EVALS = 67815918820
TOTAL_GENS = 264475110


def main():
    gif_files = glob.glob("output/**/*.gif", recursive=True)
    gif_files = sorted(gif_files)
    if not gif_files:
        print("No GIFs found.")
        return

    all_frames = []
    print(f"Found {len(gif_files)} GIFs. Loading a subset to get ~600 frames max...")

    # Target frames to pick from each GIF to spread them out over the duration
    frames_per_gif = max(1, TOTAL_FRAMES // len(gif_files))

    for gf in gif_files:
        if len(all_frames) >= 720:
            break
        try:
            # Note: imageio.mimread reads all frames
            gif_frames = imageio.mimread(gf)
            if not gif_frames:
                continue

            # Pick a few frames from this GIF
            step = max(1, len(gif_frames) // frames_per_gif)
            count = 0
            for i in range(0, len(gif_frames), step):
                if count >= frames_per_gif:
                    break
                frame_np = gif_frames[i]

                # Make sure it's RGB
                if frame_np.ndim == 2:
                    frame_np = np.stack((frame_np,) * 3, axis=-1)
                elif frame_np.shape[2] == 4:
                    frame_np = frame_np[:, :, :3]  # Drop alpha

                # Crop the top 90 pixels to remove the title
                frame_np = frame_np[90:, :, :]

                all_frames.append(frame_np)
                count += 1

                if len(all_frames) >= 720:
                    break

        except Exception as e:
            pass

    if not all_frames:
        print("No frames loaded.")
        return

    print(f"Total raw frames loaded: {len(all_frames)}")
    target_h, target_w = all_frames[0].shape[:2]

    # Standardize sizes
    std_frames = []
    for f in all_frames:
        img = Image.fromarray(f)
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.LANCZOS)
        std_frames.append(img)

    # Try to load a nice font, fallback to default if not found
    font = None
    common_fonts = [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for font_path in common_fonts:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 40)
                break
            except Exception:
                pass

    if font is None:
        font = ImageFont.load_default()
        print("Warning: Using default PIL font, text may be small.")

    print("Generating standard frames and rendering text...")

    os.makedirs("vid_frames", exist_ok=True)

    for i in range(TOTAL_FRAMES):
        progress = i / max(1, (TOTAL_FRAMES - 1))
        cur_evals = int(progress * TOTAL_EVALS)
        cur_gens = int(progress * TOTAL_GENS)

        idx_float = progress * (len(std_frames) - 1)
        idx1 = int(np.floor(idx_float))
        idx2 = int(np.ceil(idx_float))
        alpha = idx_float - idx1

        img1 = std_frames[idx1]
        img2 = std_frames[idx2]

        blended = Image.blend(img1, img2, alpha)

        # Create an RGBA overlay for text to draw transparent background
        overlay = Image.new("RGBA", blended.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        text = f"{cur_evals:,} circuits simulated\n{cur_gens:,} generations"

        try:
            try:
                left, top, right, bottom = draw.multiline_textbbox(
                    (0, 0), text, font=font, align="center"
                )
            except AttributeError:
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            tw = right - left
            th = bottom - top
        except AttributeError:
            # Fallback for older PIL versions
            try:
                tw, th = draw.multiline_textsize(text, font=font)
            except AttributeError:
                tw, th = draw.textsize(text, font=font)

        x = (target_w - tw) // 2
        y = (target_h - th) // 2
        padding_x = 40
        padding_y = 40

        draw.rectangle(
            [
                max(0, x - padding_x),
                max(0, y - padding_y),
                min(target_w, x + tw + padding_x),
                min(target_h, y + th + padding_y),
            ],
            fill=(0, 0, 0, 180),
        )
        try:
            draw.multiline_text(
                (x, y), text, font=font, fill=(255, 255, 255, 255), align="center"
            )
        except AttributeError:
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

        # Composite text overlay on blended image
        blended = blended.convert("RGBA")
        final_img = Image.alpha_composite(blended, overlay).convert("RGB")

        # Save frame
        final_img.save(f"vid_frames/frame_{i:04d}.png")

    print(f"Compiling video with ffmpeg to {OUTPUT_MP4}...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            "vid_frames/frame_%04d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            OUTPUT_MP4,
        ]
    )
    print(f"Done! Cleaning up frames...")

    # Cleanup
    for i in range(TOTAL_FRAMES):
        fpath = f"vid_frames/frame_{i:04d}.png"
        if os.path.exists(fpath):
            os.remove(fpath)
    os.rmdir("vid_frames")


if __name__ == "__main__":
    main()
