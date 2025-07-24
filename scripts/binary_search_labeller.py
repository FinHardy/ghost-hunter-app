"""
This script is used to label images in a directory using a GUI.
It initially uses a grid like sampling pattern.
Then uses a binary search algorithm to find the next best point to label.

NOTE: At the moment you have to do all the labelling in one go and you cannot continue from a checkpoint.
"""

import os
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


def logical_sort_coordinates(file_list):
    """
    Sorts a list of file paths logically based on the numeric parts of the filenames.
    To account for my sins of my dumb filenaming system I have had to create this ghastly function to sort the files as would be expected

    Args:
        file_list (list of str): List of file paths to sort.

    Returns:
        list of str: Sorted list of file paths.
    """

    def logical_sort_key(filepath):
        filename = os.path.basename(filepath)
        # Extract all numeric parts from the filename (e.g., _0_15 -> [0, 15])
        # EXCEPT boxsize_5 section (for example)
        coordinates_plus_boxsize = filename.split("_")
        numbers = coordinates_plus_boxsize[0:2]
        # Convert all extracted numbers to integers and return as a tuple
        return tuple(map(int, numbers))

    return sorted(file_list, key=logical_sort_key)


def get_files(file_path: str):
    files = []
    for root, dirs, filenames in os.walk(file_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    print(f"Found {len(files)} files.")
    return files


def do_sparse_sampling(step, width, height):
    # create zeros array of size (width, height)
    """Get values from a grid pattern."""
    sparse_indices = []
    for i in range(0, height, step):
        for j in range(0, width, step):
            sparse_indices.append((i * width) + j)
    return sparse_indices


def save_label(file_name, output_file, label):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = yaml.safe_load(f)
            if data is None:
                data = {"labels": []}
            elif "labels" not in data:
                data["labels"] = []
    else:
        data = {"labels": []}

    data["labels"].append({"file": file_name, "label": label})

    with open(output_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(
        f"Total images: {len(data['labels'])}\nSaved label '{label}' for file '{file_name}'."
    )


class LabelingApp:
    def __init__(self, master, file_path, output_file, labels_to_assign, step=20):
        self.master = master
        self.output_file = output_file
        self.master.title("File Labeling App")
        self.image_directory = file_path
        self.step = step

        self.cache = {}  # cache for binary search function

        self.labels_to_assign = labels_to_assign
        self.labels_assigned = 0
        self.binary_search_iteration = 0

        # vaildate the path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path '{file_path}' does not exist.")

        self.current_file = None
        self.current_index = 0
        files = [
            os.path.basename(f) for f in get_files(file_path) if f.endswith(".png")
        ]
        self.file_list = logical_sort_coordinates(files)

        assert self.labels_to_assign < len(files), (
            "Number of files to label is greater than the number of files available."
        )

        # get dimension of scan
        final_file = self.file_list[-1]
        self.dimensions = final_file.split("_")[0:2]
        self.height = int(self.dimensions[0]) + 1
        self.width = int(self.dimensions[1]) + 1

        # reshape the file list to an array of width, height
        self.file_list_array = np.array(self.file_list)
        self.file_list_array_reshaped = self.file_list_array.reshape(
            (self.height, self.width)
        )  # type: ignore
        print(self.file_list_array_reshaped.shape)

        # This is used to track the values and base the bayesian sampling off of
        self.sparse_array = np.zeros((self.height, self.width))

        # step through the generated sparse indices
        self.step_through_sparse_indices = 0

        # create array of initial sparse indices
        self.sparse_indices = do_sparse_sampling(
            step=self.step, width=self.width, height=self.height
        )

        print(
            f"There are {len(self.sparse_indices)} sparse indices. to label according to the step size of: {step}"
        )

        self.label_number_after_sparse_labelling = labels_to_assign - len(
            self.sparse_indices
        )

        if self.label_number_after_sparse_labelling < 0:
            raise ValueError(
                f"Label number {labels_to_assign} is less than the number of sparse indices {len(self.sparse_indices)}."
            )

        # values to make binary searching code work
        # init values
        self.non_zero_coords = []
        self.num_non_zero_coords = 0
        self.current_index = 1

        # Matplotlib heatmap setup
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, padx=20)

        # Label showing current file
        self.file_label = tk.Label(master, text="", font=("Arial", 16))
        self.file_label.pack(pady=20)

        # Image display area (self.image_label defined here)
        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

        # Buttons for labeling
        self.button_0 = tk.Button(
            master,
            text="Horizontal Polarisation",
            command=self.label_0,
            bg="red",
            font=("Arial", 14),
        )
        self.button_0.pack(side=tk.RIGHT, padx=20, pady=20)

        self.button_1 = tk.Button(
            master,
            text="Vertical Polarisation",
            command=self.label_1,
            bg="green",
            font=("Arial", 14),
        )
        self.button_1.pack(side=tk.RIGHT, padx=20, pady=20)

        self.button_2 = tk.Button(
            master,
            text="No Observable Polarisation",
            command=self.label_2,
            bg="green",
            font=("Arial", 14),
        )
        self.button_2.pack(side=tk.RIGHT, padx=20, pady=20)

        # Load first file
        self.load_file()

    def __len__(self):
        return len(self.file_list)

    def update_heatmap(self):
        self.ax.clear()

        # Show label state
        cmap = plt.cm.get_cmap("viridis", 4)  # 4 discrete label classes (0-3)
        im = self.ax.imshow(self.sparse_array, cmap=cmap, vmin=0, vmax=3)

        # Highlight the current image location
        pos = np.where(self.file_list_array_reshaped == self.current_file)
        if pos[0].size > 0 and pos[1].size > 0:
            i, j = pos[0][0], pos[1][0]
            self.ax.plot(
                j,
                i,
                marker="s",
                color="red",
                markersize=12,
                markeredgewidth=2,
                markeredgecolor="black",
            )

        self.ax.set_title("Label Grid")
        self.ax.axis("off")

        if hasattr(self, "cbar"):
            self.cbar.remove()

        # self.fig.colorbar(im, ax=self.ax, ticks=[0, 1, 2, 3], label="Label")
        self.canvas.draw()

    def binary_search_for_boundary(self):
        """
        Search for the boundary of the labeled region.
        Initially, it finds all non-zero coordinates in the sparse array.
        Then for each one of these values it find the 8 closest coordinates
        to the current coordinate.
        Then for each one of these coordinates it finds if one of them has a different label
        then returns the coordinate of the middle point and returns the coordinate
        """

        # find coordinates of all non zero values within the sparse array
        if self.current_index >= self.num_non_zero_coords:
            self.binary_search_iteration += 1
            print(f"Iteration {self.binary_search_iteration}")
            self.non_zero_coords = np.argwhere(self.sparse_array != 0)  # type: ignore
            if len(self.non_zero_coords) == 0:
                print("No labeled points found.")
                return None
            # convert to list of tuples
            self.non_zero_coords = [tuple(coord) for coord in self.non_zero_coords]  # type: ignore
            # save where you are within list
            self.num_non_zero_coords = len(self.non_zero_coords)
            self.current_index = 0

            print(f"Non-zero coords in sparse labelled array: {self.non_zero_coords}")

        # for each coordinate within the sparse array return the 8 closest coordinates
        current_coord = self.non_zero_coords[self.current_index]
        current_i, current_j = current_coord
        print(f"Current coordinate: {current_coord}")
        current_coord_value = self.sparse_array[current_i, current_j]

        # find the 8 closest coordinates within self.non_zero_coords based on ecludian distance
        ret_closest_coord = None
        min_distance = float("inf")

        max_distance_between_sparse_coords = np.sqrt(2 * self.step**2)

        for coord in self.non_zero_coords:
            if coord != current_coord:
                # calculatcurrent_coorde the ecludian distance between the two coordinates
                # TODO: This might get slow for large labelled datasets
                dist = np.linalg.norm(np.array(coord) - np.array(current_coord))
                if (
                    dist < min_distance
                    and self.sparse_array[coord] != current_coord_value
                    and dist <= max_distance_between_sparse_coords
                ):
                    mid_coord = (
                        int((current_i + coord[0]) / 2),
                        int((current_j + coord[1]) / 2),
                    )
                    if mid_coord not in self.cache:
                        ret_closest_coord = coord
                        min_distance = dist  # type: ignore

        self.current_index += 1

        if ret_closest_coord is not None:
            # find the coordinate in the middle of the two coordinates
            mid_coord = (
                int((current_i + ret_closest_coord[0]) / 2),
                int((current_j + ret_closest_coord[1]) / 2),
            )
            # cache values so no repeated labelling
            self.cache[mid_coord] = True
            print(f"Chosen coordinate: {mid_coord}")
            return mid_coord

        print("No boundary between any close coordinates.")
        return None

    def load_file(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                data = yaml.safe_load(f) or {"labels": []}
        else:
            data = {"labels": []}

        # Ensure that 'labels' is always a list
        if "labels" not in data or not isinstance(data["labels"], list):
            data["labels"] = []

        # # Get the set of already labeled files
        # labeled_files = [entry["file"] for entry in data["labels"]]

        # TODO: some way of continuing from the last file labelled
        # NOTE: Might not be possible in the middle of bindary search - or at least a pain in the ass

        if self.labels_assigned < self.labels_to_assign:
            if self.step_through_sparse_indices < len(self.sparse_indices):
                selected_index = self.sparse_indices[self.step_through_sparse_indices]
                selected_file = str(self.file_list[selected_index])
                print(selected_index, selected_file)
                self.step_through_sparse_indices += 1

            else:
                coord = None
                while coord is None:
                    coord = self.binary_search_for_boundary()

                i, j = coord
                selected_file = str(self.file_list_array_reshaped[i, j])
                self.file_label.config(
                    text=f"Current file (Binary Search): {selected_file}"
                )

            self.current_file = selected_file
            self.update_heatmap()

            self.file_label.config(text=f"Current file: {selected_file}")

            # Load and display the image
            img_path = os.path.join(self.image_directory, selected_file)
            img = Image.open(img_path).convert("L")
            # Resize the image to desired size
            desired_width = 1024
            desired_height = 1024
            img = img.resize((desired_width, desired_height))

            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo)  # type: ignore
            self.image_label.image = photo  # type: ignore

        else:
            self.file_label.config(text="All files labeled!")
            self.image_label.config(image="")
            self.button_0.config(state=tk.DISABLED)
            self.button_1.config(state=tk.DISABLED)
            # shut down app
            print(f"All {self.labels_to_assign} files labeled. Shutting down.")

            # convert sparse array to maptplotlib and save image
            # TODO: fix this vicsious hack that gets around weird bug where it ends up flipped and rotated the wrong way
            # NOTE: This bug is due to the way that the images were saved originally
            # e.g. Y_X.png or HEIGHT_WIDTH.png
            # so they will have to be flipped and rotated the other way to see properly again...
            out_image = self.sparse_array.copy()
            out_image = np.flip(out_image, axis=0)
            out_image = np.rot90(out_image, k=3)
            plt.imsave(
                f"{os.path.splitext(self.output_file)}_sparse_label_plot",
                out_image,
                cmap="viridis",
                vmin=0,
                vmax=3,
            )
            self.master.quit()

    def update_sparse_array(self, label_value):
        """
        0 for unlabelled sections
        1 for horizontal polarisation
        2 for vertical polarisation
        3 for no observable polarisation
        """
        pos = np.where(self.file_list_array_reshaped == self.current_file)
        if pos[0].size > 0 and pos[1].size > 0:
            i, j = pos[0][0], pos[1][0]
            self.sparse_array[i, j] = label_value
        else:
            raise ValueError(f"File '{self.current_file}' not found in sparse array.")

        self.labels_assigned += 1

    def label_0(self):
        if self.current_file:
            save_label(self.current_file, self.output_file, "horizontal")
            self.update_sparse_array(1)
            self.load_file()

    def label_1(self):
        if self.current_file:
            save_label(self.current_file, self.output_file, "vertical")
            self.update_sparse_array(2)
            self.load_file()

    def label_2(self):
        if self.current_file:
            save_label(self.current_file, self.output_file, "na")
            self.update_sparse_array(3)
            self.load_file()


def label(png_images_file_path, labels_output_file, labels_to_assign, step=20):
    """
    Function to label files in a directory using the LabelingApp GUI.
    """
    root = tk.Tk()
    app = LabelingApp(
        root,
        png_images_file_path,
        output_file=labels_output_file,
        labels_to_assign=labels_to_assign,
        step=step,
    )
    print(f"Loaded {len(app)} files.")
    root.mainloop()


if __name__ == "__main__":
    # Create the GUI
    root = tk.Tk()
    app = LabelingApp(
        root,
        "your_data_here",
        output_file="50_labels_opt_strat_75_step.yaml",
        labels_to_assign=50,
        step=75,
    )
    print(f"Loaded {len(app)} files.")
    root.mainloop()
