import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib
matplotlib.use('TkAgg')  # <<< 이 줄을 추가
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

def generate_data(num_points):
    vectors = []
    x = np.random.normal(2, 2, num_points) + 10  # ⟵ 동작 코드 구현 필요 (2점)                            # ⟵ 동작 코드 구현 필요 (2점)
    noise = 2 * np.random.normal(0, 3, num_points)  # ⟵ 동작 코드 구현 필요 (2점)                       # ⟵ 동작 코드 구현 필요 (2점)
    y = 5 * x + noise  # ⟵ 동작 코드 구현 필요 (2점)
    vectors.append([ x, y ])  # ⟵ 동작 코드 구현 필요 (2점)
    x_data = np.array([v[0] for v in vectors])
    y_data = np.array([v[1] for v in vectors])
    return x_data, y_data

def plot_data(x_data, y_data):
    plt.plot(x_data, y_data, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 25])
    plt.ylim([0, 100])
    plt.title('Raw Data')
    plt.show()


def train_model(x_data, y_data, learning_rate=0.0015, steps=10):
    # TODO W, b 변수 선언
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # ⟵ 동작 코드 구현 필요 (2점)
    b = tf.Variable(tf.zeros([1]))  # ⟵ 동작 코드 구현 필요 (2점)
    # 모델 정의: y_pred  W * x + b
    y_pred = W * x_data + b # ⟵ 동작 코드 구현 필요 (2점)
    # 손실 함수 정의: 평균 제곱 오차
    loss = tf.reduce_mean(tf.square(y_pred- y_data))  # <-- 동작 코드 구현 필요 (2점)
    # 옵티마이저 정의
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    weight_list = []
    bias_list = []
    loss_list = []

    for step in range(steps):
        sess.run(train)
        current_w, current_b, current_loss = sess.run([W, b, loss])
        print(f"Step {step}: W  {current_w}, b = {current_b}, loss = {current_loss}")
        weight_list.append(current_w)
        bias_list.append(current_b)
        loss_list.append(current_loss)
        # 학습 결과 시각화
        plt.plot(x_data, current_w * x_data + current_b, 'bo')
        plt.plot(x_data, y_data, 'ro')
        plt.title(f'Step {step}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    return weight_list, bias_list, loss_list

if __name__ == '__main__':
    num_points = 50
    x_data, y_data = generate_data(num_points)
    plot_data(x_data, y_data)
    w_vals, b_vals, losses = train_model(x_data, y_data)
    print("W 변화:", w_vals)
    print("b 변화:", b_vals)
    print("Loss 변화:", losses)