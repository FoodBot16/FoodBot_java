
/*
MLP neural network in Java
by Phil Brierley
www.philbrierley.com
http://www.philbrierley.com/main.html?code/index.html&code/codeleft.html
This code may be freely used and modified at will

Tanh hidden neurons
Linear output neuron

To include an input bias create an
extra input in the training data
and set to 1

Routines included:

calcNet()
WeightChangesHO()
WeightChangesIH()
initWeights()
initData()
tanh(double x)
displayResults()
calcOverallError()

compiled and tested on
Symantec Cafe Lite

*/

import java.lang.Math;

public class JavaMLP {

	// 사용자가 수정할 수 있는 매개변수 설정(user defineable variables)
	public static int numEpochs = 500; // Epoch(number of training cycles)
	public static int numInputs = 3; // 입력 뉴런의 수+bias 포함(number of inputs - this includes the input bias)
	public static int numHidden = 4; // 히든층 뉴런의 수 (number of hidden units)
	public static int numPatterns = 4; // 패턴 수(number of training patterns)
	public static double LR_IH = 0.7; // Input-Hidden learning rate
	public static double LR_HO = 0.07; // Hidden-Output learning rate

	//  변수(process variables)
	public static int patNum;
	public static double errThisPat;
	public static double outPred;
	public static double RMSerror;

	// training data
	public static double[][] trainInputs = new double[numPatterns][numInputs];
	public static double[] trainOutput = new double[numPatterns];

	// 히든층 뉴런의 출력 배열(the outputs of the hidden neurons)
	public static double[] hiddenVal = new double[numHidden];

	// the weights
	public static double[][] weightsIH = new double[numInputs][numHidden];
	public static double[] weightsHO = new double[numHidden];

	// ==============================================================
	// ********** THIS IS THE MAIN PROGRAM **************************
	// ==============================================================

	public static void main(String[] args) {

		// 가중치 초기화(initiate the weights)
		initWeights();

		// 데이터 입력(load in the data)
		initData();

		// 학습(train the network)
		for (int j = 0; j <= numEpochs; j++) {

			for (int i = 0; i < numPatterns; i++) {

				// 무작위로 데이터 선택(select a pattern at random)
				patNum = (int) ((Math.random() * numPatterns) - 0.001);

				// calculate the current network output
				// and error for this pattern
				// 학습 후 출력과 오차 계산
				calcNet();

				// 가중치 조정(change network weights)
				WeightChangesHO();
				WeightChangesIH();
			}

			// display the overall network error
			// after each epoch
			// Epoch마다 전체 오차 출력
			calcOverallError();
			System.out.println("epoch = " + j + "  RMS Error = " + RMSerror);

		}

		// training has finished
		// display the results
		// 학습 종료 후 결과 출력
		displayResults();

	}

	// ============================================================
	// ********** END OF THE MAIN PROGRAM **************************
	// =============================================================

	// 신경망 학습
	public static void calcNet() {
		// calculate the outputs of the hidden neurons
		// the hidden neurons are tanh
		// 히든층의 출력 계산. 활성화함수는 tanh
		for (int i = 0; i < numHidden; i++) {
			hiddenVal[i] = 0.0;

			for (int j = 0; j < numInputs; j++)
				hiddenVal[i] = hiddenVal[i] + (trainInputs[patNum][j] * weightsIH[j][i]);

			hiddenVal[i] = tanh(hiddenVal[i]);
		}

		// calculate the output of the network
		// the output neuron is linear
		// 
		outPred = 0.0;

		for (int i = 0; i < numHidden; i++)
			outPred = outPred + hiddenVal[i] * weightsHO[i];

		// 오차 계산(calculate the error)
		errThisPat = outPred - trainOutput[patNum];
	}

	// Hidden-Output 가중치 조정
	public static void WeightChangesHO()	{
		// Hidden-Output 가중치 조정(adjust the weights hidden-output)
		for (int k = 0; k < numHidden; k++) {
			double weightChange = LR_HO * errThisPat * hiddenVal[k];
			weightsHO[k] = weightsHO[k] - weightChange;

			// 출력층에서의 가중치 정규화(regularization on the output weights)
			if (weightsHO[k] < -5)
				weightsHO[k] = -5;
			else if (weightsHO[k] > 5)
				weightsHO[k] = 5;
		}
	}

	// Input-Hidden 가중치 조정
	public static void WeightChangesIH()	{
		// Input-Hidden 가중치 조정(adjust the weights input-hidden)
		for (int i = 0; i < numHidden; i++) {
			for (int k = 0; k < numInputs; k++) {
				double x = 1 - (hiddenVal[i] * hiddenVal[i]);
				x = x * weightsHO[i] * errThisPat * LR_IH;
				x = x * trainInputs[patNum][k];
				double weightChange = x;
				weightsIH[k][i] = weightsIH[k][i] - weightChange;
			}
		}
	}

	//  가중치 초기화
	public static void initWeights() {

		for (int j = 0; j < numHidden; j++) {
			weightsHO[j] = (Math.random() - 0.5) / 2;
			for (int i = 0; i < numInputs; i++)
				weightsIH[i][j] = (Math.random() - 0.5) / 5;
		}
	}

	// 데이터 입력
	public static void initData() {

		System.out.println("initializing data");

		// the data here is the XOR data
		// it has been rescaled to the range
		// [-1][1]
		// an extra input valued 1 is also added
		// to act as the bias

		// 예시는 XOR이므로 다음과 같은 데이터를 입력함
		// 입력 데이터에 bias 값으로 1이 추가됨
		trainInputs[0][0] = 1;
		trainInputs[0][1] = -1;
		trainInputs[0][2] = 1;// bias
		trainOutput[0] = 1;

		trainInputs[1][0] = -1;
		trainInputs[1][1] = 1;
		trainInputs[1][2] = 1;// bias
		trainOutput[1] = 1;

		trainInputs[2][0] = 1;
		trainInputs[2][1] = 1;
		trainInputs[2][2] = 1;// bias
		trainOutput[2] = -1;

		trainInputs[3][0] = -1;
		trainInputs[3][1] = -1;
		trainInputs[3][2] = 1;// bias
		trainOutput[3] = -1;

	}

	// 활성화 함수 Tanh
	public static double tanh(double x) {
		if (x > 20)
			return 1;
		else if (x < -20)
			return -1;
		else {
			double a = Math.exp(x);
			double b = Math.exp(-x);
			return (a - b) / (a + b);
		}
	}
	
	// 활성화 함수 Sigmoid
	public static double sig(double x){
		return 1/(1+Math.exp(-x));
	}

	//  결과 출력
	public static void displayResults() {
		for (int i = 0; i < numPatterns; i++) {
			patNum = i;
			calcNet();
			System.out.println("pat = " + (patNum + 1) + " actual = " + trainOutput[patNum]
					+ " neural model = " + outPred);
		}
	}

	// 전체 오차 계산
	public static void calcOverallError() {
		RMSerror = 0.0;
		for (int i = 0; i < numPatterns; i++) {
			patNum = i;
			calcNet();
			RMSerror = RMSerror + (errThisPat * errThisPat);
		}
		RMSerror = RMSerror / numPatterns;
		RMSerror = java.lang.Math.sqrt(RMSerror);
	}
}