<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<meta charset="UTF-8">
<link href="bootstrap/css/bootstrap.min.css" rel="stylesheet"
	media="screen">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="bootstrap/css/bootstrap-responsive.css" rel="stylesheet">
<title>Title</title>


<style>
div {
	border: 0px solid red;
}

body {
	padding-top: 60px;
}
</style>
<title>회귀 학습 ㅎㅎㅎㅎㅎ</title>
</head>
<body>
	<div class="navbar navbar-inverse navbar-fixed-top">
		<div class="navbar-inner">
			<div class="container">
				<button type="button" class="btn btn-navbar" data-toggle="collapse"
					data-target=".nav-collapse">
					<span class="icon-bar"></span> <span class="icon-bar"></span> <span
						class="icon-bar"></span>
				</button>
				<a class="brand" href="#">1조를벌자</a>
				<div class="nav-collapse collapse">
					<ul class="nav">
						<li class="home"><a href="#">Home</a></li>
						<li class="chat"><a href="#about">Chat</a></li>
						<li class="ml"><a href="#ml">ML</a></li>
						<li class="taste"><a href="#">MyTaste</a></li>
						<li><a href="#contact">About</a></li>
						<li class="faq"><a href="#contact">FAQ</a></li>
					</ul>
				</div>
				<!--/.nav-collapse -->
			</div>
		</div>
	</div>


	<div class="container-fluid">

		<div class="row-fluid hero-unit">
			<h2>1조를 벌자</h2>
		</div>

		<div class="row-fluid">
			<div class="span3">채팅하는곳</div>
			<div class="span7">

				<script>
					result();
				</script>
			</div>
			<div class="row-fluid">
				<div class="span12 offset3 result">
					<%
						if (request.getAttribute("result") != null) {
							out.print(request.getAttribute("result"));

							request.removeAttribute("result");
						}
					%>

				</div>
			</div>
		</div>
		<!--
        <div class=" hero-unit navbar-fixed-bottom">
            <p>문의번호 : 010-123-4567</p>
        </div>
            -->
	</div>
	<script src="js/jquery.js"></script>
	<script src="bootstrap/js/bootstrap.min.js"></script>

	<script>
	
	
	$(".taste").click(function() {
		$.ajax({
			url : './taste.html',
			success : function(data) {
				$(".span7").html(data);
			}
		})
	});
	$(".chat").click(function() {
		$.ajax({
			url : './chat.html',
			success : function(data) {
				$(".span7").html(data);
				$(".result").html("");
			}
		})
	});
		$(".brand").click(function() {
			$.ajax({
				url : './home.html',
				success : function(data) {
					$(".span7").html(data);
					$(".result").html("");
				}
			})
		});
		$(".home").click(function() {
			$.ajax({
				url : './home.jsp',
				success : function(data) {
					$(".span7").html(data);
					$(".result").html("");
				}
			})
		});
		$(".faq").click(function() {
			$.ajax({
				url : './ex2.html',
				success : function(data) {
					$(".span7").html(data);
				}
			})
		});
		$(".ml").click(function() {
			$.ajax({
				url : './ml.html',
				success : function(data) {
					$(".span7").html(data);
				}
			})
		});
		$("*").click(function() {
	<%request.removeAttribute("result");%>
		});
	</script>
</body>
</html>