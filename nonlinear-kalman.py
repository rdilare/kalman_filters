import numpy as np
import pygame, sys
from numpy import sin,cos

import drawSvg as draw

pygame.init()


def info(d,datalabel,x,y,textsize,textcolor,datacolor):
	d.append(draw.Text(datalabel, textsize, x,y, center=False, fill=textcolor,stroke=textcolor, stroke_width=0))
	d.append(draw.Line(x-50, y+textsize/4, x-5, y+textsize/4, fill=datacolor,stroke=datacolor, stroke_width=2))

def draw_bot(d,pose,radius):
	r = radius
	x,y,ang = pose

	p1 = (x-(r-8)*sin(ang),y+(r-8)*cos(ang))
	q1 = (x+(r-8)*sin(ang),y-(r-8)*cos(ang))
	p2 = (x+r*cos(ang),y+r*sin(ang))
	q2 = (x-r*cos(ang),y-r*sin(ang))

	d.append(draw.Circle(x, y, r, fill="#140741", stroke_width=1.5, stroke="#ffff00"))
	d.append(draw.Line(p1[0], p1[1], q1[0], q1[1], fill="#ffff00", stroke_width=1, stroke="#ffff00"))
	d.append(draw.Line(p2[0], p2[1], q2[0], q2[1], fill="#00ff00", stroke_width=2, stroke="#00ff00"))

def draw_frame(kalman_history, real_history, measurement_history):
	w,h = 600,400
	d = draw.Drawing(w, h, origin=(0,0))
	d.setRenderSize(h=400)
	# d.append(draw.Circle(0, y, 1, fill='lime'))
	d.append(draw.Rectangle(0,0,w, h, fill='#140741'))
	info(d,"kalman estimation",450,350,15,'#ffffab','blue')
	info(d,"sensor reading",450,330,15,'#ffffab','red')
	info(d,"actual position",450,310,15,'#ffffab','#ffff00')
	


	x,y,ang = real_history[-1]
	draw_bot(d,(x,y,ang),20)


	for i in kalman_history:
		# pygame.draw.circle(screen,(0,255,255), (int(i[0]),int(i[1])), 2)
		d.append(draw.Circle(int(i[0]), int(i[1]), 1.5, fill='#0000ff'))

	for i in real_history:
		# pygame.draw.circle(screen,(255,255,0), (int(i[0]),int(i[1])), 2)
		d.append(draw.Circle(int(i[0]), int(i[1]), 1.5, fill='#ffff00', opacity=0.7))

	for i in measurement_history:
		# pygame.draw.circle(screen,(255,255,0), (int(i[0]),int(i[1])), 2)
		d.append(draw.Circle(int(i[0]), int(i[1]), 1.5, fill='red'))

	return d






def drawPlus(surf,pose,size,thickness,color):
	r = size
	x,y,ang = pose
	p1 = (x-r*sin(ang),y+r*cos(ang))
	q1 = (x+r*sin(ang),y-r*cos(ang))
	p2 = (x+r*cos(ang),y+r*sin(ang))
	q2 = (x-r*cos(ang),y-r*sin(ang))
	pygame.draw.line(surf,color,(int(p1[0]),int(p1[1])),(int(q1[0]),int(q1[1])),int(thickness))
	pygame.draw.line(surf,color,(int(p2[0]),int(p2[1])),(int(q2[0]),int(q2[1])),int(thickness))



def mult(arr):
	l = len(arr)
	temp = arr[-1]
	for i in range(l-1):
		temp = np.dot(arr[-2-i],temp)
	return temp



def kalman(x,A,B,C,P,R,Q,u,z):

	x_bar = np.dot(A,x) + np.dot(B,u)
	P_bar = np.dot(np.dot(A,np.linalg.inv(P)),np.transpose(A)) + R

	K = mult([P_bar,np.transpose(C),np.linalg.inv(mult([C,P_bar,np.transpose(C)])+Q)])

	x_hat = x_bar+mult([K,(z-mult([C,x_bar]))])
	P_hat = P_bar - mult([K,C,P_bar])

	return x_hat, P_hat


def motion_model(X,t):
	x = X[:]
	x[0] = X[0]+X[3]*cos(X[2])*t
	x[1] = X[1]+X[3]*sin(X[2])*t
	x[2] = X[2]+X[4]*t
	x[3] = X[3]
	x[4] = X[4]

	return x




x = 250
y = 200
theta = 0
v = 0
thetadot = 0

vdot = 0
theta_doubledot = 0


X = [x,y,theta,v,thetadot]
u = [vdot,theta_doubledot]
t=.05		#delta time-step in sec.


def Amat(t,X):
	a = [[1, 0, -X[3]*t*sin(X[2]), t*cos(X[2]), 0],
		[0, 1, X[3]*t*cos(X[2]), t*sin(X[2]), 0],
		[0, 0, 1, 0, t],
		[0, 0, 0, 1, 0],
		[0, 0, 0, 0, 1]]

	return a

B = [[0, 0],
	 [0, 0],
	 [0, 0],
	 [0, 0],
	 [0, 0]]

C = [[1,0,0,0,0],    #mesurement matrix
	 [0,1,0,0,0],    
	 [0,0,0,0,0],    
	 [0,0,0,0,0],    
	 [0,0,0,0,0]]


P = [[100,0,0,0,0],		# initial state covariance
	 [0,100,0,0,0],
	 [0,0,10,0,0],
	 [0,0,0,100,0],
	 [0,0,0,0,10]] 	

R = [[10,0,0,0,0],		# model noise covariance
	 [0,10,0,0,0],
	 [0,0,0.1,0,0],
	 [0,0,0,5,0],
	 [0,0,0,0,.01]] 		 

# Q = [[10,0],		# measurement noise covariance
# 	 [0,.001]]	

Q = [[50,0,0,0,0],		# measurement noise covariance
	 [0,50,0,0,0],
	 [0,0,.2,0,0],
	 [0,0,0,10,0],
	 [0,0,0,0,.1]] 



time = 0


w,h = 600,400
screen = pygame.display.set_mode((w,h))
clock = pygame.time.Clock()


x0_kalman = np.array(X)+np.array([300,0,0,0,0])   # initializing the kalman filter with initial guess


kalman_history = []
real_history = []
measurement_history=[]

running = True

with draw.animate_video('kalman(non-linear).gif', draw_frame, duration=1/15) as anim:

	while running:
		clock.tick(20)

		for ev in pygame.event.get():
			if ev.type == pygame.QUIT:
				running = False
			if ev.type == pygame.KEYDOWN:
				if ev.key == pygame.K_ESCAPE:
					running = False

		v = 100
		thetadot = 1.3
		time+=t

		if time%10 > 4:
			thetadot *=-1

		A =  Amat(t,X)

		# x_real = np.dot(A,X) + np.dot(B,u) + np.random.randn(5)*.1
		x_real = motion_model(X,t) + np.random.randn(5)*[1,1,.02,5,.01]

		z = mult([C,x_real[:]]) + np.random.randn(5)*[3,3,.03,7,.05]
		X = [x_real[0],x_real[1],x_real[2],v,thetadot]

		x_kalman,P = kalman(x0_kalman,A,B,C,P,R,Q,u,z)
		x0_kalman=x_kalman[:]

		kalman_history.append(x_kalman[0:3])
		real_history.append(x_real[0:3])
		measurement_history.append(z)

		screen.fill((0,0,0))
		pygame.draw.circle(screen,(255,255,0),(int(X[0]),int(X[1])),20)			#Robot
		drawPlus(screen,(x_kalman[0], x_kalman[1], 0),5,3,(0,150,255))
		drawPlus(screen,(X[0], X[1], X[2]),18,2,(5,5,5))

		for i in kalman_history:
			pygame.draw.circle(screen,(0,255,255), (int(i[0]),int(i[1])), 2)	#kalman estimation

		for i in real_history:
			pygame.draw.circle(screen,(255,255,0), (int(i[0]),int(i[1])), 2)	#actual position

		for i in measurement_history:
			pygame.draw.circle(screen,(255,0,0), (int(i[0]),int(i[1])), 2)		#measurement

		if len(kalman_history)>200:
			kalman_history.pop(0)
			real_history.pop(0)
			measurement_history.pop(0)

		pygame.display.update()


		# anim.draw_frame(kalman_history, real_history, measurement_history)

	sys.exit()
