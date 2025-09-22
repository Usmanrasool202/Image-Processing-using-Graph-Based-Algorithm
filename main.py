import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import matplotlib.pyplot as plt
import mst_edge_detection
import graph_cut_segmentation

class GraphTheoryImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Theory Image Processor")
        self.root.geometry("400x400")
        self.image_path = None

        # Title
        self.label = tk.Label(root, text="Graph Theory Image Processor", font=("Arial", 16))
        self.label.pack(pady=10)

        # Upload Image Button
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image, width=30, height=2)
        self.upload_btn.pack(pady=10)

        # MST Edge Detection Button (disabled until image is uploaded)
        self.mst_btn = tk.Button(root, text="Show MST Edge Detection", command=self.run_mst, width=30, height=2, state="disabled")
        self.mst_btn.pack(pady=10)

        # Graph Cut Segmentation Button (disabled until image is uploaded)
        self.gc_btn = tk.Button(root, text="Show Graph Cut Segmentation", command=self.run_graphcut, width=30, height=2, state="disabled")
        self.gc_btn.pack(pady=10)

        # Exit Button
        self.exit_btn = tk.Button(root, text="Exit", command=root.quit, width=30, height=2)
        self.exit_btn.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.mst_btn.config(state="normal")
            self.gc_btn.config(state="normal")
            self.label.config(text=f"Selected: {file_path.split('/')[-1]}")
        else:
            messagebox.showwarning("No Image", "Please select a valid image.")

    def run_mst(self):
        if self.image_path:
            gray_img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            edge_img = mst_edge_detection.mst_edge_detection(self.image_path, gray_img.shape)
            plt.figure("MST Edge Detection")
            plt.imshow(edge_img, cmap='gray')
            plt.title("MST Edge Detection")
            plt.axis('off')
            plt.show()

    def run_graphcut(self):
        if self.image_path:
            original_img, segmented_img = graph_cut_segmentation.grabcut_segmentation(self.image_path)
            plt.figure("Graph Cut Segmentation")
            plt.imshow(segmented_img)
            plt.title("Graph Cut Segmentation")
            plt.axis('off')
            plt.show()

def main():
    root = tk.Tk()
    app = GraphTheoryImageProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
