import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# --- Configuration ---
USE_DUMMY_DATA = False

# Layout Configuration (Wide format)
# FIGURE_WIDTH = 18
# FIGURE_HEIGHT = 6
FIGURE_WIDTH = 5
FIGURE_HEIGHT = 9
FONT_FAMILY = "serif"
FS_MAIN_TITLE = 20
FS_SECTION_TITLE = 16
FS_SUB_TITLE = 14
FS_LABEL = 12
FS_TOKEN_LABEL = 16  # Font size for the new token labels
FS_FOOTNOTE = 10

GROUPS = [
    {
        "title": "Cyberpunk",
        "main": "outputs/highlight/cyber_main.png",
        "subs": ["outputs/highlight/cyber_1.png", "outputs/highlight/cyber_2.png", "outputs/highlight/cyber_3.png"],
    },
    {
        "title": "Celestial",
        "main": "outputs/highlight/horse_main.png",
        "subs": ["outputs/highlight/horse_1.png", "outputs/highlight/horse_2.png", "outputs/highlight/horse_3.png"],
    },
    {
        "title": "Stone behemoth",
        "main": "outputs/highlight/stone_golem_main.png",
        "subs": [
            "outputs/highlight/stone_golem_1.png",
            "outputs/highlight/stone_golem_2.png",
            "outputs/highlight/stone_golem_3.png",
        ],
    },
    {
        "title": "Leviathan",
        "main": "outputs/highlight/leviatan_main.png",
        "subs": [
            "outputs/highlight/leviatan_1.png",
            "outputs/highlight/leviatan_2.png",
            "outputs/highlight/leviatan_3.png",
        ],
    },
]
COMPARISON_IMAGE = "outputs/intreventions/highlight/highlight_d5_c17_pca_fix.png"


# --- Helper: Dummy Data Generation ---
def create_dummy_image(name, color, size=(512, 512)):
    img = Image.new("RGB", size, color=color)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("times.ttf", 60)
    except:
        font = ImageFont.load_default()
    d.rectangle([0, 0, size[0] - 1, size[1] - 1], outline="white", width=4)
    bbox = d.textbbox((0, 0), name, font=font)
    x = (size[0] - (bbox[2] - bbox[0])) / 2
    y = (size[1] - (bbox[3] - bbox[1])) / 2
    d.text((x, y), name, fill=(255, 255, 255), font=font)
    return img


if USE_DUMMY_DATA:
    print("Generating dummy images...")
    colors = ["#C0392B", "#D35400", "#2980B9"]
    for i, group in enumerate(GROUPS):
        create_dummy_image(group["title"], colors[i], (600, 600)).save(group["main"])
        for j, sub in enumerate(group["subs"]):
            create_dummy_image(f"{j+1}", colors[i], (256, 256)).save(sub)
    create_dummy_image("Comparison\nGrid", "#7F8C8D", (800, 800)).save(COMPARISON_IMAGE)


def load_img(path):
    try:
        img = Image.open(path)
        rgb_im = img.convert("RGB")
        rgb_im.save(path[:-3] + "jpg")
        return Image.open(path[:-3] + "jpg")
    except:
        return create_dummy_image("?", "#95A5A6")


# --- CORE LOGIC: Combine Images in Memory ---
def combine_group_images(main_path, sub_paths):
    main_img = load_img(main_path)
    sub_imgs = [load_img(p) for p in sub_paths]
    w_main, h_main = main_img.size
    target_sub_w = w_main // 3

    resized_subs = []
    for sub in sub_imgs:
        w_s, h_s = sub.size
        scale = target_sub_w / w_s
        target_sub_h = int(h_s * scale)
        resized_subs.append(sub.resize((target_sub_w, target_sub_h), Image.Resampling.LANCZOS))

    row_height = resized_subs[0].size[1]
    total_height = h_main + row_height
    combined = Image.new("RGB", (w_main, total_height))
    combined.paste(main_img, (0, 0))
    current_x = 0
    for sub in resized_subs:
        combined.paste(sub, (current_x, h_main))
        current_x += sub.size[0]
    return combined


# --- Helper: Fancy Arrow ---
def add_arrow(ax, start, end, **kwargs):
    defaults = {"arrowstyle": "-|>", "mutation_scale": 30, "color": "black", "linewidth": 2, "clip_on": False}
    defaults.update(kwargs)
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, **defaults))


# --- Main Script ---
def create_aligned_figure():
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.serif": ["Times New Roman"]})
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    # gs_outer = gridspec.GridSpec(1, 3, width_ratios=[0.19, 1.8, 0.96], wspace=0.04, figure=fig)
    gs_outer = gridspec.GridSpec(1, 1, wspace=0.0, hspace=0.0, figure=fig)

    # ==========================
    # 1. INPUT (Left)
    # ==========================
    # ax_input = fig.add_subplot(gs_outer[0])
    # ax_input.axis('off')
    # ax_input.text(0.5, 0.65, 'Input:\n"Monster"', ha='center', va='center', fontsize=FS_SECTION_TITLE, fontweight='bold')
    # add_arrow(ax_input, (0.2, 0.5), (0.9, 0.5), linewidth=3, mutation_scale=40)
    # ax_input.text(0.5, 0.35, "Unsupervised\nEmbedding Space\nDecomposition", ha='center', va='center', fontsize=FS_LABEL, style='italic')

    # ==========================
    # 2. MIDDLE (The Directions)
    # ==========================
    # fig.text(0.50, 0.95, 'Discovered Directions for "Monster"', ha='center', va='center', fontsize=FS_MAIN_TITLE)
    # gs_middle_cols = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[1], wspace=0.00)
    # gs_middle_cols = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[0], wspace=0.0, hspace=0.1)

    # for i, group in enumerate(GROUPS):
    #     ax = fig.add_subplot(gs_middle_cols[i//2, i%2])
    #     combined_img = combine_group_images(group['main'], group['subs'])
    #     ax.imshow(combined_img, aspect='equal')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.axis('off')
    #     ax.set_title(group['title'], pad=10, fontsize=FS_SUB_TITLE)

    # ==========================
    # 3. RIGHT (Comparison)
    # ==========================
    # ax_right = fig.add_subplot(gs_outer[2])
    ax_right = fig.add_subplot(gs_outer[0])
    ax_right.imshow(load_img(COMPARISON_IMAGE), aspect="equal")
    ax_right.axis("off")

    # # Title
    # ax_right.text(0.5, 1.15, "Token-wise controlability", ha='center', va='bottom',
    #               fontsize=FS_SECTION_TITLE, transform=ax_right.transAxes)

    # --- COORDINATES (Moved Closer) ---
    ARROW_H_Y = 1.03  # Horizontal Arrow Y position (Closer to image top)
    ARROW_V_X = -0.03  # Vertical Arrow X position (Closer to image left)

    # --- Horizontal Arrow & Labels ---
    # 1. "Austin FX4" Label at end of arrow path
    t_austin = ax_right.text(
        1.0,
        ARROW_H_Y,
        "Oriental Shorthair",
        ha="right",
        va="center",
        fontsize=FS_SUB_TITLE,
        transform=ax_right.transAxes,
    )

    # Calculate arrow end based on text position
    bb = t_austin.get_window_extent(renderer=fig.canvas.get_renderer())
    bb_ax = ax_right.transAxes.inverted().transform(bb)
    arrow_end_x = bb_ax[0, 0]

    # Draw Horizontal Arrow
    add_arrow(ax_right, (0.0, ARROW_H_Y), (arrow_end_x + 0.02, ARROW_H_Y))

    # Add "token=car" label ABOVE the horizontal arrow
    ax_right.text(
        0.5,
        ARROW_H_Y + 0.015,
        "token=cat",
        ha="center",
        va="bottom",
        fontsize=FS_TOKEN_LABEL,
        style="italic",
        color="#444444",
        transform=ax_right.transAxes,
    )

    # --- Vertical Arrow & Labels ---
    # 1. "Mechanical" Label at bottom of arrow path (shifted down slightly for alignment)
    t_mech = ax_right.text(
        ARROW_V_X + 0.02,
        0.0,
        "Labrador",
        ha="right",
        va="bottom",
        rotation=90,
        fontsize=FS_SUB_TITLE,
        transform=ax_right.transAxes,
    )

    # Calculate arrow end based on text position
    bb = t_mech.get_window_extent(renderer=fig.canvas.get_renderer())
    bb_ax = ax_right.transAxes.inverted().transform(bb)
    arrow_end_y = bb_ax[1, 1]

    # Draw Vertical Arrow (Top to Bottom)
    add_arrow(ax_right, (ARROW_V_X, 1.0), (ARROW_V_X, arrow_end_y + 0.08))

    # Add "token=monster" label TO THE LEFT of the vertical arrow (Rotated 90)
    ax_right.text(
        ARROW_V_X - 0.015,
        0.5,
        "token=dog",
        ha="right",
        va="center",
        rotation=90,
        fontsize=FS_TOKEN_LABEL,
        style="italic",
        color="#444444",
        transform=ax_right.transAxes,
    )

    # # Footnote
    # fig.text(0.5, 0.08, "* The labels are generated by Gemini 3 post-discovery",
    #          ha='center', va='center', fontsize=FS_FOOTNOTE, style='italic', color='#555555')

    plt.subplots_adjust(left=0.05, right=0.92, top=0.80, bottom=0.1)

    # Save
    plt.savefig("publication_figure_aligned.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("publication_figure_aligned.png", dpi=300, bbox_inches="tight")
    print("Saved to publication_figure_aligned.pdf")


if __name__ == "__main__":
    create_aligned_figure()
    if USE_DUMMY_DATA:
        import glob

        for f in glob.glob("*.png"):
            try:
                os.remove(f)
            except:
                pass
