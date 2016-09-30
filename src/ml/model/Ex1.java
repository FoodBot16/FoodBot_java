package ml.model;

import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.List;
import ml.vo.*;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import oracle.jrockit.jfr.jdkevents.ThrowableTracer;

public class Ex1 {

	private static final double epsilon = Double.MIN_VALUE;
	private static double mu[];
	private static double sigma[];

	private static Attribute test = null;
	private static Double result = null;

	public void getAttribute(Attribute t) {

	}

	public double run(double[] t) {
		List<Attribute> dataX = nomalization(loadDataX());
		List<Double> dataY = loadDataY();
		double[] user = new double[t.length+1];
		double alpha = 0.01;
		int maxIterations = 10000;

		List<Double> finalTheta = singleVarGradientDescent(dataX, dataY, initialTheta(), alpha, maxIterations);
		// System.out.printf("theta0 = %f, theta1 = %f", finalTheta.getX(),
		// finalTheta.getY());

		for (int i = 0; i < finalTheta.size(); i++) {
			System.out.printf("%d theta = %.5f ", i, finalTheta.get(i));
			System.out.println();
		}
		
		user[0] = 1;
		for(int i = 1; i<t.length+1; i++)
			user[i] = t[i-1];

		System.out.println("cost : " + hypothesis(dataX, dataY, finalTheta));

		double r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0;
		double result = 0;

		for(int i = 1; i<t.length+1; i++)
			System.out.print(user[i]+", ");
		System.out.println();
		double[] in1 = { 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0 };
		double[] in2 = { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
		double[] in3 = { 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0 };
		double[] in4 = { 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0 };
		double[] in5 = { 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0 };
		double[] in6 = { 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0 };
		System.out.println(in1.length);
		System.out.println(finalTheta.size());

		// 노멀라이즈는 x0 는 포함하면 안되서 i가 1부터시작함
		for (int i = 1; i < in1.length; i++) {

/*			in1[i] = (in1[i] - mu[i]) / sigma[i];
			in2[i] = (in2[i] - mu[i]) / sigma[i];
			in3[i] = (in3[i] - mu[i]) / sigma[i];
			in4[i] = (in4[i] - mu[i]) / sigma[i];
			in5[i] = (in5[i] - mu[i]) / sigma[i];
			in6[i] = (in6[i] - mu[i]) / sigma[i];*/
			
			user[i] = (user[i] - mu[i]) / sigma[i];
		}
		
		for (int i = 0; i < finalTheta.size(); i++) {
/*			r1 += finalTheta.get(i) * in1[i];
			r2 += finalTheta.get(i) * in2[i];
			r3 += finalTheta.get(i) * in3[i];
			r4 += finalTheta.get(i) * in4[i];
			r5 += finalTheta.get(i) * in5[i];
			r6 += finalTheta.get(i) * in6[i];*/
			result += finalTheta.get(i) * user[i];
		}
		/*
		 * // 분석 결과값을 반올림 처리 r1 = Math.round(r1); r2 = Math.round(r2); r3 =
		 * Math.round(r3); r4 = Math.round(r4); r5 = Math.round(r5);
		 */
		System.out.println(r1 + " " + r2 + " " + r3 + " " + r4 + " " + r5 + " " + r6);
		
		System.out.println("사용자 =>" + result);
		return result;

	}

	private List<Attribute> nomalization(List<Attribute> dataX) {

		int attribute_num = dataX.get(0).getAttribute().size();
		mu = new double[attribute_num];
		sigma = new double[attribute_num];
		double sum_tmp = 0.0;

		List<Attribute> new_dataX = new ArrayList();
		int num = dataX.size() * dataX.get(0).getAttribute().size();
		int input_num = dataX.size();
		for (int i = 0; i < dataX.get(0).getAttribute().size(); i++) {
			sum_tmp = 0.0;
			for (int j = 0; j < dataX.size(); j++) {
				sum_tmp += dataX.get(j).getAttribute().get(i);
			}
			mu[i] = sum_tmp / input_num;
		}
		for (int i = 0; i < dataX.get(0).getAttribute().size(); i++) {
			sum_tmp = 0.0;
			for (int j = 0; j < dataX.size(); j++) {
				sum_tmp += Math.pow((dataX.get(j).getAttribute().get(i) - mu[i]), 2);
			}
			sigma[i] = sum_tmp / (input_num - 1);
			sigma[i] = Math.sqrt(sigma[i]);
		}
		for (int i = 0; i < dataX.size(); i++) {
			ArrayList<Double> tmp = new ArrayList<>();
			for (int j = 0; j < dataX.get(i).getAttribute().size(); j++) {
				tmp.add((dataX.get(i).getAttribute().get(j) - mu[j]) / sigma[j]);
			}
			new_dataX.add(new Attribute().setBias(1.0).setTaste_1000(tmp.get(1)) // 17
					.setTaste_2000(tmp.get(2)).setTaste_4000(tmp.get(3)).setTaste_5000(tmp.get(4))
					.setTaste_6000(tmp.get(5)).setTaste_7000(tmp.get(6)).setTaste_9000(tmp.get(7))
					.setMeat_1000(tmp.get(8)).setMeat_3000(tmp.get(9)).setFlour_1000(tmp.get(10))
					.setFlour_2000(tmp.get(11)));
		}
		// for(int i=0 ; i<12;i++) {
		//
		// System.out.printf("%.3f ",mu[i]);
		// }
		// System.out.println();
		// for(int i=0 ; i<12;i++) {
		// System.out.printf("%.3f ",sigma[i]);
		// }
		// System.out.println();
		// for(int i=0 ; i<dataX.size() ; i++) {
		// for(int j=0 ;j<dataX.get(i).getAttribute().size() ; j++) {
		// System.out.print(new_dataX.get(i).getAttribute().get(j) + " ");
		// }
		// System.out.println();
		// }
		return new_dataX;
	}

	private List<Double> initialTheta() {
		List<Double> list = new ArrayList<Double>();
		list.add(1.0);
		for (int i = 0; i < 11; i++) {
			list.add(0.0);
		}
		return list;
	}

	private List<Double> singleVarGradientDescent(List<Attribute> dataX, List<Double> dataY, List<Double> theta,
			double alpha, int maxIterations) {

		int m = dataX.size();
		for (int i = 0; i < maxIterations; i++) {

			ArrayList<Double> theta_tmp = new ArrayList();
			ArrayList<Double> htheta = new ArrayList();

			for (int j = 0; j < dataX.size(); j++) {
				double tmp = 0.0;
				for (int k = 0; k < dataX.get(j).getAttribute().size(); k++) {

					tmp += dataX.get(j).getAttribute().get(k) * theta.get(k);
				}
				htheta.add(tmp);
			}

			for (int j = 0; j < 12; j++) {
				theta_tmp.add(theta.get(j) - alpha / m * gradientSub(htheta, dataY, dataX, j));
			}
			for (int j = 0; j < 12; j++) {
				theta.set(j, theta_tmp.get(j));
			}
			hypothesis(dataX, dataY, theta);
		}
		return theta;
	}

	private double gradientSub(ArrayList<Double> htheta, List<Double> dataY, List<Attribute> dataX, int idx) {
		double sum = 0.0;
		for (int i = 0; i < dataY.size(); i++) {
			sum += (htheta.get(i) - dataY.get(i)) * dataX.get(i).getAttribute().get(idx);
		}
		return sum;
	}

	private double hypothesis(List<Attribute> x, List<Double> y, List<Double> theta) {
		double j = 0.0;
		double tmp = 0.0;
		for (int i = 0; i < x.size(); i++) {
			tmp = 0.0;
			for (int k = 0; k < x.get(i).getAttribute().size(); k++) {
				tmp += (x.get(i).getAttribute().get(k) * theta.get(k));
			}
			j += (tmp - y.get(i)) * (tmp - y.get(i));

		}
		j = j / (2 * x.size());
		return j;
	}

	private List<Attribute> loadDataX() {
		List<Attribute> data = new ArrayList<>();

		data.add(new Attribute().setInitialData("1,1,1,0,0,0,0,0,1,0,0"));
		data.add(new Attribute().setInitialData("1,1,0,0,0,0,0,0,0,0,1"));
		data.add(new Attribute().setInitialData("0,0,0,0,1,0,1,1,0,0,0"));
		data.add(new Attribute().setInitialData("0,0,1,0,1,0,0,0,0,1,0"));
		data.add(new Attribute().setInitialData("0,0,0,1,0,0,1,0,1,0,0"));
		data.add(new Attribute().setInitialData("0,0,0,1,0,1,0,0,0,1,0"));

		return data;
	}

	private List<Double> loadDataY() {
		List<Double> data = new ArrayList<>();

		data.add(1.0);
		data.add(2.0);
		data.add(3.0);
		data.add(4.0);
		data.add(5.0);
		data.add(6.0);

		return data;
	}
}