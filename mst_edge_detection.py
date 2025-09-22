import cv2
import numpy as np
import networkx as nx

def mst_edge_detection(image_path, img_shape):
    # Step 1: Read grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    rows, cols = img_shape

    # Step 2: Create graph G
    G = nx.Graph()
    
    # Add nodes and edges with weights (intensity differences)
    for i in range(rows):
        for j in range(cols):
            current = i * cols + j
            G.add_node(current)

            # 4-connected neighbors (right and down)
            if j + 1 < cols:
                right = i * cols + (j + 1)
                weight = abs(int(img[i, j]) - int(img[i, j + 1]))
                G.add_edge(current, right, weight=weight)

            if i + 1 < rows:
                down = (i + 1) * cols + j
                weight = abs(int(img[i, j]) - int(img[i + 1, j]))
                G.add_edge(current, down, weight=weight)

    # Step 3: Compute MST using Kruskal’s algorithm
    mst = nx.minimum_spanning_tree(G, weight='weight')

    # Step 4: Highlight high-weight edges as edges
    edge_image = np.zeros((rows, cols), dtype=np.uint8)

    for u, v, data in mst.edges(data=True):
        weight = data['weight']
        if weight > 15:  # Threshold, adjust as needed
            # Get pixel coordinates from node index
            x1, y1 = divmod(u, cols)
            x2, y2 = divmod(v, cols)
            edge_image[x1, y1] = 255
            edge_image[x2, y2] = 255

    return edge_image
