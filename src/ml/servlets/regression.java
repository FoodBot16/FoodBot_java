package ml.servlets;

import ml.model.Ex1;
import ml.vo.Attribute;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Servlet implementation class regression
 */
@WebServlet("/regression")
public class regression extends HttpServlet {
	private static final long serialVersionUID = 1L;

	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		double[] tmp = new double[11];
		
		for(int i = 0; i<tmp.length;i++){
			String tt = request.getParameter("x"+(i+1));
			if(tt != null){
				tmp[i] = Double.parseDouble(tt);
			}
			else{
				tmp[i] = 0;
			}
		}
/*		new Attribute().setX1(tmp.get(0)).setX2(tmp.get(1))
		.setX3(tmp.get(2)).setX4(tmp.get(3)).setX5(tmp.get(4))
		.setX6(tmp.get(5)).setX7(tmp.get(6)).setX8(tmp.get(7));*/
		
		double result = new Ex1().run(tmp);
		int realresult = (int)Math.round(result);
		String[] food = {"제육볶음","떡볶이","치킨","파스타","탕수육","냉면"};
		request.setAttribute("result", food[realresult-1]+" 를 추천해드립니다.");
		RequestDispatcher rd = request.getRequestDispatcher("index.jsp");
		rd.forward(request, response);	
	}	
}
