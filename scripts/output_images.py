from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
plt.rc('text', usetex=True)


def combine_pdfs_2x2(pdf_files, subtitles, output_png, dpi=300):
    """
    Open four PDF files, render the first page of each as an image,
    arrange them in a 2x2 grid with subtitles, and save as a high-resolution PNG.

    pdf_files : list of str  -> four PDF paths
    subtitles : list of str  -> four subtitles
    output_png: str         -> output file path
    dpi       : int         -> output DPI (default 300)
    """
    if len(pdf_files) != 4 or len(subtitles) != 4:
        raise ValueError("Need exactly 4 PDF files and 4 subtitles.")

    # convert first page of each PDF to PIL image at desired dpi
    images = [convert_from_path("Images/"+p, dpi=dpi, first_page=1, last_page=1)[0]
              for p in pdf_files]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=dpi)
    canvas = FigureCanvas(fig)

    for ax, img, subtitle in zip(axes.flatten(), images, subtitles):
        ax.imshow(img)
        # subtitle can contain <b>…</b> and <i>…</i> for selective styling
        ax.text(
            0.5, -0.05, subtitle,
            ha='center', va='top',
            transform=ax.transAxes,
            fontsize=12
        )
        ax.axis('off')

    plt.tight_layout()
    canvas.print_figure(output_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


for metric in ["operationnal_prc", "prc", "roc", "calibration"]:
    for chu in [True, False]:
        pdfs = [
            f"{"chu_" if chu else ''}test_full_{metric}.pdf",
            f"{"chu_" if chu else ''}test_no_hypo_{metric}.pdf",
            f"{"chu_" if chu else ''}test_no_inter_{metric}.pdf",
            f"{"chu_" if chu else ''}test_no_hypo_no_inter_{metric}.pdf"
        ]
        subtitles = [r"\textbf{(a)} \textit{Full} test set", r"\textbf{(b)} \textit{no_hypo} test set",
                     r"\textbf{(c)} \textit{no_inter} test set", r"\textbf{(d)} \textit{no_hypo_no_inter} test set"]
        combine_pdfs_2x2(pdfs, subtitles, f"output_word/{'chu_' if chu else 'vdb'}_{metric}.png")


for chu in [True, False]:
    pdfs = [
        f"{"chu_" if chu else ''}test_no_hypo_no_inter_{metric}.pdf" for metric in ["operationnal_prc", "prc", "roc", "calibration"]
    ]
    subtitles = [r"\textbf{(a)} Operationnal PRC", r"\textbf{(b)} ROC",
                 r"\textbf{(c)} Standard PRC", r"\textbf{(d)} Calibration curve"]
    combine_pdfs_2x2(pdfs, subtitles, f"output_word/{'chu_' if chu else 'vdb'}_all_metrics.png")
