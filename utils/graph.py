# graph.py
"""
Graph definitions including a new 'isl' layout intended for
MediaPipe upper-body + both hands:

 - pose: 33 landmarks (indices 0..32)
 - left hand: 21 landmarks  (indices 33..53)
 - right hand: 21 landmarks (indices 54..74)

Total nodes: 33 + 21 + 21 = 75

The pose connectivity uses a reasonable upper-body subset of the MediaPipe
pose topology (should be adapted if you use a different indexing).
Hand connectivity follows the standard 21-point hand topology (root + fingers).
"""
import numpy as np
from typing import List, Tuple

def edge2mat(link: List[Tuple[int,int]], num_node: int) -> np.ndarray:
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A: np.ndarray) -> np.ndarray:
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w), dtype=A.dtype)
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_hop_distance(num_node: int, edge: List[Tuple[int,int]], max_hop: int = 1) -> np.ndarray:
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_adjacency_matrix(A: np.ndarray) -> np.ndarray:
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    degs_inv_sqrt[np.isinf(degs_inv_sqrt)] = 0.0
    norm_degs_matrix = np.eye(len(node_degrees), dtype=A.dtype) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

def k_adjacency(A: np.ndarray, k: int, with_self: bool = False, self_factor: int = 1) -> np.ndarray:
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

class Graph:
    """
    Graph class used by ST-GCN and related models.

    Supported layouts (existing + new):
      - openpose, ntu-rgb+d, ipn, shrec*, ...
      - mediapipe_upper  -> 33 pose + 21 left hand + 21 right hand (75 nodes)
    """
    def __init__(self, layout: str = 'openpose', strategy: str = 'uniform', max_hop: int = 1, dilation: int = 1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return f"Graph(layout={getattr(self,'layout',None)}, num_node={self.num_node})"

    def get_edge(self, layout: str):
        self.layout = layout
        # existing layouts (keep compatibility)
        if layout == 'ipn' or layout == 'briareo':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
            return

        # ... keep other layouts if needed (shrec*, openpose, etc.)
        # For brevity only a few are retained below; extend as needed.
        if layout == 'openpose':
            # classic OpenPose 18 joints connectivity (subset)
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (0,1),(1,2),(2,3),(3,4),      # right arm / face
                (0,5),(5,6),(6,7),(7,8),      # left arm / face
                (0,9),(9,10),(10,11),(11,12), # torso / hips
                (9,13),(13,14),(14,15),(11,16),(16,17)
            ]
            neighbor_link = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
            return

        # New layout: MediaPipe upper body + both hands (75 nodes)
        if layout == 'isl':
            # Indices:
            #  - pose: 0..32  (33 nodes)
            #  - left hand: 33..53  (21 nodes)
            #  - right hand: 54..74 (21 nodes)
            self.num_node = 75
            # --------------------------------------------------
            # Self links
            # --------------------------------------------------
            self_link = [(i, i) for i in range(self.num_node)]
            #connect wrist from pose with hands
            # --------------------------------------------------
            # Pose graph (MediaPipe Pose, upper-body focused)
            # --------------------------------------------------
            pose_edges = [
                # shoulders & arms
                (11,12),
                (11,13),(13,15),   # left shoulder → elbow → wrist
                (12,14),(14,16),   # right shoulder → elbow → wrist

                # torso
                (11,23),(12,24),
                (23,24),

                # head (compact, stable)
                (0,1),(1,2),(2,3),(3,7),
                (0,4),(4,5),(5,6),(6,8),
                (9,10),
                (0,11),(0,12),
            ]

            # --------------------------------------------------
            # Feet (weak, minimal influence)
            # --------------------------------------------------
            feet_edges = [
                # left leg
                (23,25),(25,27),(27,29),(29,31),
                # right leg
                (24,26),(26,28),(28,30),(30,32),
            ]

            pose_edges += feet_edges

            # Make pose bidirectional
            pose_edges = pose_edges + [(j,i) for (i,j) in pose_edges]

            # --------------------------------------------------
            # Hand skeleton (MediaPipe Hands, 21 joints)
            # --------------------------------------------------
            hand_pairs_21 = [

                (0,1),(1,2),(2,3),(3,4),        # thumb
                (0,5),(5,6),(6,7),(7,8),        # index
                (0,9),(9,10),(10,11),(11,12),   # middle
                (0,13),(13,14),(14,15),(15,16), # ring
                (0,17),(17,18),(18,19),(19,20)  # pinky
            ]

            # Left hand
            left_offset = 33
            left_hand_edges = [(a+left_offset, b+left_offset) for (a,b) in hand_pairs_21]
            #connect joints from left hand with pose joints
            left_hand_edges += [(b,a) for (a,b) in left_hand_edges]

            # Right hand
            right_offset = 54
            right_hand_edges = [(a+right_offset, b+right_offset) for (a,b) in hand_pairs_21]
            right_hand_edges += [(b,a) for (a,b) in right_hand_edges]

            # --------------------------------------------------
            # Hand ↔ Wrist connections (critical for ISL)
            # --------------------------------------------------
            left_wrist_pose = 15   # MediaPipe left wrist
            right_wrist_pose = 16 # MediaPipe right wrist

            connect_hands = [
                (left_wrist_pose, left_offset + 0),
                (left_offset + 0, left_wrist_pose),
                (right_wrist_pose, right_offset + 0),
                (right_offset + 0, right_wrist_pose),
            ]

            # --------------------------------------------------
            # Final edge list
            # --------------------------------------------------
            neighbor_link = []
            neighbor_link += pose_edges
            neighbor_link += left_hand_edges
            neighbor_link += right_hand_edges
            neighbor_link += connect_hands

            self.edge = self_link + neighbor_link

            # --------------------------------------------------
            # Spatial center (important for ST-GCN)
            # --------------------------------------------------
            self.center = 11  # left shoulder
            return

        # If layout not recognized, raise
        raise ValueError(f"Do Not Exist This Layout: {layout}")

    def get_adjacency(self, strategy: str):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
            return

        if strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
            return

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            return

        raise ValueError(f"Do Not Exist This Strategy: {strategy}")
