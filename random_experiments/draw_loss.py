import plotly.graph_objects as go
import tensorflow as tf
from loss.sat import softplus_mixed_loss, softplus_loss

if __name__ == '__main__':
    clauses = tf.RaggedTensor.from_tensor([[1, -2], [-1, 2], [-2, -1], [1, 2]])

    v1_all = [w / 2 for w in range(-40, 40, 1)]
    v2_all = [b / 2 for b in range(-40, 40, 1)]

    loss_landscape = []
    for v2 in v2_all:
        tmp = []
        for v1 in v1_all:
            loss = tf.reduce_sum(softplus_loss(tf.constant([float(v1), float(v2)]), clauses)).numpy()
            tmp.append(loss)
        loss_landscape.append(tmp)

    fig = go.Figure(data=[go.Surface(z=loss_landscape, x=v2_all, y=v1_all)])

    fig.update_layout(title='(-1 or 2) and (-1 or -2) and (1 or -2) and (-1 or 2) and (1 or -2) loss', autosize=True,
                      scene=dict(
                          xaxis_title='v1',
                          yaxis_title='v2',
                          zaxis_title='Loss'
                      ),
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()
