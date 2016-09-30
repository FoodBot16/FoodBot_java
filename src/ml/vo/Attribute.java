package ml.vo;

import java.util.ArrayList;
import java.util.List;

public class Attribute {
	private double bias;
	
	private double taste_1000;
	private double taste_2000;
	private double taste_3000;
	private double taste_4000;
	private double taste_5000;
	private double taste_6000;
	private double taste_7000;
	private double taste_8000;
	private double taste_9000;
	private double meat_1000;
	private double meat_2000;
	private double meat_3000;
	
	private double flour_1000;
	

	private double flour_2000;
	

	
	
	private int num = 12;
	private double y;
	
	private List<Double> list;
	public Attribute() {
		this.bias = 1;
		this.taste_1000 = 0;
		this.taste_2000 = 0;
		//this.taste_3000 = 0;
		this.taste_4000 = 0;
		this.taste_5000 = 0;
		this.taste_6000 = 0;
		//this.taste_8000 = 0;
		this.taste_9000 = 0;
		this.meat_1000 = 0;
		//this.meat_2000 = 0;
		this.meat_3000 = 0;
		
		this.flour_1000 = 0;
		this.flour_2000 = 0;
		
		this.y = 0;
		
		list = new ArrayList<Double>();
	}
	
	public List<Double> getAttribute() {
		list =null;
		
		list = new ArrayList<Double>();
		list.add(bias);
		list.add(taste_1000);
		list.add(taste_2000);
		//list.add(taste_3000);
		list.add(taste_4000);
		list.add(taste_5000);
		list.add(taste_6000);
		list.add(taste_7000);
		//list.add(taste_8000);
		list.add(taste_9000);
		list.add(meat_1000);
		//list.add(meat_2000);
		list.add(meat_3000);
		list.add(flour_1000);
		list.add(flour_2000);
		
		return list;
	}
	public double getTaste_7000() {
		return taste_7000;
	}

	public Attribute setTaste_7000(double taste_7000) {
		this.taste_7000 = taste_7000;return this;
	}

	public double getFlour_1000() {
		return flour_1000;
	}

	public Attribute setFlour_1000(double flour_1000) {
		this.flour_1000 = flour_1000;return this;
	}

	public double getFlour_2000() {
		return flour_2000;
	}

	public Attribute setFlour_2000(double flour_2000) {
		this.flour_2000 = flour_2000;return this;
	}
	public double getBias() {
		return bias;
	}

	public Attribute setBias(double bias) {
		this.bias = bias;return this;
	}

	public double getTaste_1000() {
		return taste_1000;
	}

	public Attribute setTaste_1000(double taste_1000) {
		this.taste_1000 = taste_1000;return this;
	}

	public double getTaste_2000() {
		return taste_2000;
	}

	public Attribute setTaste_2000(double taste_2000) {
		this.taste_2000 = taste_2000;return this;
	}

	public double getTaste_3000() {
		return taste_3000;
	}

	public Attribute setTaste_3000(double taste_3000) {
		this.taste_3000 = taste_3000;return this;
	}

	public double getTaste_4000() {
		return taste_4000;
	}

	public Attribute setTaste_4000(double taste_4000) {
		this.taste_4000 = taste_4000;return this;
	}

	public double getTaste_5000() {
		return taste_5000;
	}

	public Attribute setTaste_5000(double taste_5000) {
		this.taste_5000 = taste_5000;return this;
	}

	public double getTaste_6000() {
		return taste_6000;
	}

	public Attribute setTaste_6000(double taste_6000) {
		this.taste_6000 = taste_6000;return this;
	}

	public double getTaste_8000() {
		return taste_8000;
	}

	public Attribute setTaste_8000(double taste_8000) {
		this.taste_8000 = taste_8000;return this;
	}

	public double getTaste_9000() {
		return taste_9000;
	}

	public Attribute setTaste_9000(double taste_9000) {
		this.taste_9000 = taste_9000;return this;
	}

	public double getMeat_1000() {
		return meat_1000;
	}

	public Attribute setMeat_1000(double meat_1000) {
		this.meat_1000 = meat_1000;return this;
	}

	public double getMeat_2000() {
		return meat_2000;
	}

	public Attribute setMeat_2000(double meat_2000) {
		this.meat_2000 = meat_2000;return this;
	}

	public double getMeat_3000() {
		return meat_3000;
	}

	public Attribute setMeat_3000(double meat_3000) {
		this.meat_3000 = meat_3000;return this;
	}

	public int getNum() {
		return num;
	}

	public Attribute setNum(int num) {
		this.num = num;return this;
	}

	public double getY() {
		return y;
	}

	public Attribute setY(double y) {
		this.y = y;
		return this;
	}

	public List<Double> getList() {
		return list;
	}

	public Attribute setList(List<Double> list) {
		this.list = list;
		return this;
	}

	public Attribute setInitialData(String data) {
		String[] strArr = data.split(",");
		this.setTaste_1000(Double.parseDouble(strArr[0]));
		this.setTaste_2000(Double.parseDouble(strArr[1]));
		//this.setTaste_3000(Double.parseDouble(strArr[2]));
		this.setTaste_4000(Double.parseDouble(strArr[2]));
		this.setTaste_5000(Double.parseDouble(strArr[3]));
		this.setTaste_6000(Double.parseDouble(strArr[4]));
		this.setTaste_7000(Double.parseDouble(strArr[5]));
		//this.setTaste_8000(Double.parseDouble(strArr[6]));
		this.setTaste_9000(Double.parseDouble(strArr[6]));
		this.setMeat_1000(Double.parseDouble(strArr[7]));
		//this.setMeat_2000(Double.parseDouble(strArr[9]));
		this.setMeat_3000(Double.parseDouble(strArr[8]));
		this.setFlour_1000(Double.parseDouble(strArr[9]));
		this.setFlour_2000(Double.parseDouble(strArr[10]));
	
		
		return this;
	}
}