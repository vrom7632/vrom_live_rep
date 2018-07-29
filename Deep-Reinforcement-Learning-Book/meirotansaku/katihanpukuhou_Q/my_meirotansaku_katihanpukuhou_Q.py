import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def init_byouga(plt):
    """ 迷路の初期描画"""
    # 迷路を描画
    fig = plt.figure(figsize=(5, 5))
    
    ax = plt.gca()

    # 壁の描画
    plt.plot([1,1], [0,1], color='red', linewidth=2)
    plt.plot([1,2], [2,2], color='red', linewidth=2)
    plt.plot([2,2], [2,1], color='red', linewidth=2)
    plt.plot([2,3], [1,1], color='red', linewidth=2)

    # 状態を示す文字の描画
    plt.text(0.5, 2.5, 'S0', size=14, ha='center')
    plt.text(1.5, 2.5, 'S1', size=14, ha='center')
    plt.text(2.5, 2.5, 'S2', size=14, ha='center')
    plt.text(0.5, 1.5, 'S3', size=14, ha='center')
    plt.text(1.5, 1.5, 'S4', size=14, ha='center')
    plt.text(2.5, 1.5, 'S5', size=14, ha='center')
    plt.text(0.5, 0.5, 'S6', size=14, ha='center')
    plt.text(1.5, 0.5, 'S7', size=14, ha='center')
    plt.text(2.5, 0.5, 'S8', size=14, ha='center')
    plt.text(0.5, 2.3, 'START', ha='center')
    plt.text(2.5, 0.3, 'GOAL', ha='center')

    # 描画範囲の設定
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # 現在地S0に緑丸を描画する
    line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)
    return fig,line

def init_theta(np):
    """ thetaの初期値の決定 """
    # 行は状態0～7、列は移動方向で↑、→、↓、←を表す
    return np.array([[np.nan, 1, 1, np.nan],  # snp.nan
                        [np.nan, 1, np.nan, 1],  # s1
                        [np.nan, np.nan, 1, 1],  # s2
                        [1, 1, 1, np.nan],  # s3
                        [np.nan, np.nan, 1, 1],  # s4
                        [1, np.nan, np.nan, np.nan],  # s5
                        [1, np.nan, np.nan, np.nan],  # s6
                        [1, 1, np.nan, np.nan],  # s7、※s8はゴールなので、方策はなし
                        ],dtype=float)

def simple_convert_into_pi_from_theta(theta):
    """方策パラメータtheta_0をランダム方策piに変換する関数の定義"""
    """単純に割合を計算する"""
    theta = np.array(list(map(lambda th:th/np.nansum(th),theta))) # 割合の計算
    theta = np.nan_to_num(theta)  # nanを0に変換
    return theta

def get_action(s, Q, epsilon, pi_0):
    """次の行動を決定する関数"""
    # action:[up, right, down, left]
    action = [0, 1, 2, 3]
    # 行動を決める
    if np.random.rand() < epsilon:
        # εの確率でランダムに動く
        next_action = np.random.choice(action, p=pi_0[s])
    else:
        # Qの最大値の行動を採用する
        next_action  = action[np.nanargmax(Q[s])]

    return next_action

# def get_s_next(s, a, Q, epsilon, pi_0):
def get_s_next(s, a):
    """次の状態を決定する関数"""
    # action:[up, right, down, left]
    direction = [-3, 1, 3, -1]
    s_next = s + direction[a]
    return  s_next

def q_learning(s, a, r, s_next, Q, eta, gamma):
    """Q学習による行動価値関数Qの更新"""
    if s_next == 8:  # ゴールした場合
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next]) - Q[s, a])

    return Q

def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    """Q_learningで迷路を解く関数の定義、状態と行動の履歴および更新したQを出力"""
    s = 0  # スタート地点
    a_next = get_action(s, Q, epsilon, pi)  # 初期の行動
    s_a_history = [[0, np.nan]]  # エージェントの移動を記録するリスト

    while True:  # ゴールするまでループ
        a = a_next  # 行動更新

        s_a_history[-1][1] = a
        # 現在の状態（つまり一番最後なのでindex=-1）に行動を代入

        s_next = get_s_next(s, a)
        # 次の状態を格納

        s_a_history.append([s_next, np.nan])
        # 次の状態を代入。行動はまだ分からないのでnanにしておく

        # 報酬を与え,　次の行動を求めます
        if s_next == 8:
            r = 1  # ゴールにたどり着いたなら報酬を与える
            a_next = np.nan
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi)
            # 次の行動a_nextを求めます。

        # 価値関数を更新
        Q = q_learning(s, a, r, s_next, Q, eta, gamma)

        # 終了判定
        if s_next == 8:  # ゴール地点なら終了
            break
        else:
            s = s_next

    return [s_a_history, Q]

def agent_animation(s_a_history, fig, line):
    """初期化関数とフレームごとの描画関数を用いて動画を作成する"""
    def init_background():
        """背景画像の初期化"""
        line.set_data([], [])
        return (line,)

    def animate(i):
        """フレームごとの描画内容"""
        state = s_a_history[i][0]  # 現在の場所を描く
        x = (state % 3) + 0.5  # 状態のx座標は、3で割った余り+0.5
        y = 2.5 - int(state / 3)  # y座標は3で割った商を2.5から引く
        line.set_data(x, y)
        return (line,)

    return animation.FuncAnimation(fig, animate, init_func=init_background, frames=len(
        s_a_history), interval=400, repeat=False)

if __name__ == "__main__":
    # 迷路の初期描画
    fig,line = init_byouga(plt)

    # 探索可能位置の初期化
    theta_0 = init_theta(np)

    # ランダム行動方策pi_0を求める
    pi_0 = simple_convert_into_pi_from_theta(theta_0)    
    
    # 初期の行動価値関数Qを設定
    [a, b] = theta_0.shape  # 行と列の数をa, bに格納
    Q = np.random.rand(a, b) * theta_0
    # * theta0をすることで要素ごとに掛け算をし、Qの壁方向の値がnanになる
    eta = 0.1  # 学習率
    gamma = 0.9 # 時間割引率
    epsilon = 0.5  # ε-greedy法の初期値
    v = np.nanmax(Q, axis=1)  # 状態ごとに価値の最大値を求める
    episode_count = 100
    for i in range(0,episode_count):
        print("episode:{}".format(i))

        # ε-greedyの値を少しずつ小さくする
        epsilon = epsilon / 2
        # Q_learningで迷路を解き、移動した履歴と更新したQを求める
        [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)

        # 状態価値の変化
        new_v = np.nanmax(Q, axis=1)  # 状態ごとに価値の最大値を求める
        print(np.sum(np.abs(new_v - v)))  # 状態価値の変化を出力
        print("Q:{}".format(Q))
        v = new_v
        print("step:{}".format(len(s_a_history) - 1))
        # if i == 0:
        #     ani = agent_animation(s_a_history, fig, line)
        #     ani.save("meirotansaku_katihanpukuhou_Q_1_gakushumae.mp4", writer="ffmpeg")
        # elif (i + 1) % 10 == 0:
        #     ani = agent_animation(s_a_history, fig, line)
        #     ani.save("meirotansaku_katihanpukuhou_Q_2_gakushuchu_{}.mp4".format(i), writer="ffmpeg")
            
    # print("count:{}".format(count))
    # 迷路探索のアニメーション描画
    # ani = agent_animation(s_a_history, fig, line)
    # ani.save("meirotansaku_katihanpukuhou_Q_3_gakushugo.mp4", writer="ffmpeg")
