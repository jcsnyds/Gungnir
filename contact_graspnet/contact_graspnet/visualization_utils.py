import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors

import mesh_utils


def show_image(rgb, segmap):
    """Save RGB image and segmentation overlay as interactive HTML."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1)

    if rgb is not None:
        fig.add_trace(go.Image(z=rgb))

    if segmap is not None:
        fig.add_trace(go.Heatmap(
            z=segmap,
            colorscale="Rainbow",
            opacity=0.5,
            showscale=False,
        ))

    fig.update_layout(title="RGB Image and Segmentation")
    fig.write_html("output/contact_graspnet_image.html")


def plot_coordinates(t, r, fig, length=0.2):
    """Plot coordinate axes using Plotly."""
    colors = ["red", "green", "blue"]

    for i, color in enumerate(colors):
        end = t + length * r[:, i]

        fig.add_trace(go.Scatter3d(
            x=[t[0], end[0]],
            y=[t[1], end[1]],
            z=[t[2], end[2]],
            mode="lines",
            line=dict(color=color, width=5),
            showlegend=False,
        ))


def draw_pc_with_colors(pc, fig, pc_colors=None):
    """Add a point cloud to a Plotly figure."""
    if pc is None or len(pc) == 0:
        return

    marker = dict(size=2, color="gray")

    if pc_colors is not None:
        colors = np.asarray(pc_colors)

        if colors.max() <= 1.0:
            colors = colors * 255

        colors = np.clip(colors, 0, 255).astype(np.uint8)

        marker["color"] = [
            f"rgb({r},{g},{b})" for r, g, b in colors
        ]

    fig.add_trace(go.Scatter3d(
        x=pc[:, 0],
        y=pc[:, 1],
        z=pc[:, 2],
        mode="markers",
        marker=marker,
        name="Point cloud",
    ))


def draw_grasps(
    grasps,
    cam_pose,
    gripper_openings,
    fig,
    color="green",
    colors=None,
):
    """Add predicted gripper wireframes to a Plotly figure."""
    gripper = mesh_utils.create_gripper("panda")
    control = gripper.get_control_point_tensor(
        1, False, convex_hull=False
    ).squeeze()

    midpoint = 0.5 * (control[1] + control[2])

    base_lines = np.array([
        np.zeros(3),
        midpoint,
        control[1],
        control[3],
        control[1],
        control[2],
        control[4],
    ])

    for i, (grasp, opening) in enumerate(
        zip(grasps, gripper_openings)
    ):
        points = base_lines.copy()
        points[2:, 0] = (
            np.sign(base_lines[2:, 0]) * opening / 2
        )

        points = points @ grasp[:3, :3].T + grasp[:3, 3]

        points = np.column_stack([
            points, np.ones(len(points))
        ])
        points = (points @ cam_pose.T)[:, :3]

        line_color = color if colors is None else colors[i]

        # Connect the gripper's seven control points.
        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (4, 5),
            (4, 6),
        ]

        for a, b in connections:
            segment = points[[a, b]]

            fig.add_trace(go.Scatter3d(
                x=segment[:, 0],
                y=segment[:, 1],
                z=segment[:, 2],
                mode="lines",
                line=dict(color=line_color, width=4),
                showlegend=False,
            ))



def visualize_grasps(
    full_pc,
    pred_grasps_cam,
    scores,
    plot_opencv_cam=False,
    pc_colors=None,
    gripper_openings=None,
    gripper_width=0.08,
):
    """Visualize grasps, highlighting the highest-scoring predictions."""
    print("Generating grasp visualization...")

    fig = go.Figure()
    draw_pc_with_colors(full_pc, fig, pc_colors)

    if plot_opencv_cam:
        plot_coordinates(np.zeros(3), np.eye(3), fig)

    # Collect valid scores to establish thresholds across all grasps.
    all_scores = []
    for k, grasps in pred_grasps_cam.items():
        if grasps is None or len(grasps) == 0:
            continue

        if scores is not None and k in scores:
            grasp_scores = np.asarray(scores[k]).reshape(-1)
            all_scores.extend(
                grasp_scores[np.isfinite(grasp_scores)].tolist()
            )

    if all_scores:
        score_70 = np.percentile(all_scores, 70)
        score_90 = np.percentile(all_scores, 90)
    else:
        score_70 = score_90 = None

    for k, grasps in pred_grasps_cam.items():
        if grasps is None or len(grasps) == 0:
            continue

        openings = (
            np.full(len(grasps), gripper_width)
            if gripper_openings is None
            else gripper_openings[k]
        )

        grasp_colors = []

        grasp_scores = (
            np.asarray(scores[k]).reshape(-1)
            if scores is not None and k in scores
            else np.array([])
        )

        for i in range(len(grasps)):
            if i >= len(grasp_scores) or not np.isfinite(grasp_scores[i]):
                grasp_colors.append("gray")
                continue

            score = grasp_scores[i]

            if score_90 is not None and score >= score_90:
                grasp_colors.append("gold")
            elif score_70 is not None and score >= score_70:
                grasp_colors.append("limegreen")
            else:
                grasp_colors.append("rgba(130,130,130,0.35)")

        draw_grasps(
            grasps,
            np.eye(4),
            openings,
            fig,
            colors=grasp_colors,
        )

    fig.update_layout(
        title="Contact-GraspNet Predicted Grasps (Gold = Top 10%, Green = Top 30%)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
    )

    output_file = "output/contact_graspnet_grasps.html"
    fig.write_html(output_file, auto_open=False)
    print(f"Saved grasp visualization to {output_file}")
