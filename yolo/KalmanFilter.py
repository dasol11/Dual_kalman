import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, std_proc, std_meas):
        """

        :param dt: sampling time (time for 1 cycle)
        :param std_proc: 프로세스의 표준편차
        :param std_meas: 측정의 표준 편차

        """

        # Define sampling time
        self.dt = dt


        # Intial State
        self.x = np.matrix([
                            [0],
                            [0],
                            [0],
                            [0]
                                        ])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])



        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Initial Process Noise Covariance
        self.Q = (std_proc ** 2) * np.eye(4)

        # Initial Measurement Noise Covariance
        self.R = (std_meas**2) * np.eye(2)

        # Initial Covariance Matrix
        self.P = 100 * np.eye(self.A.shape[1])

    def predict(self):
        
        # x_k =Ax_(k-1) + Bu_(k-1)
        self.x = np.dot(self.A, self.x)


        # P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

    
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))

        I = np.eye(self.H.shape[1])

        # 공분산 행렬 업데이트식
        self.P = (I - (K * self.H)) * self.P
        
        return self.x[0:2]

