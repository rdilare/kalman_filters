import numpy as np
import pygame, sys
from numpy import sin,cos

import drawSvg as draw

pygame.init()


def info(d,datalabel,x,y,textsize,textcolor,datacolor):
	d.append(draw.Text(datalabel, textsize, x,y, center=False, fill=textcolor,stroke=textcolor, stroke_width=0))
	d.append(draw.Line(x-50, y+textsize/4, x-5, y+textsize/4, fill=datacolor,stroke=datacolor, stroke_width=2))

def draw_frame(kalman_history, real_history, measurement_history):
	w,h = 600,400
	d = draw.Drawing(w, h, origin=(0,0))
	d.setRenderSize(h=400)
	# d.append(draw.Circle(0, y, 1, fill='lime'))
	d.append(draw.Rectangle(0,0,w, h, fill='#140741'))
	info(d,"kalman estimation",450,350,15,'#ffffab','blue')
	info(d,"sensor reading",450,330,15,'#ffffab','red')
	info(d,"actual position",450,310,15,'#ffffab','#ffff00')
	


	x,y = real_history[-1]
	d.append(draw.Circle(x, y, 20, fill="#140741", stroke_width=1.5, stroke="#ffff00"))
	d.append(draw.Line(x-18, y, x+18, y, fill="#ffff00", stroke_width=1, stroke="#ffff00"))
	d.append(draw.Line(x, y-18, x, y+18, fill="#ffff00", stroke_width=1, stroke="#ffff00"))


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




def drawPlus(surf,pos,size,thickness,color):
	r = size
	x,y = pos
	pygame.draw.line(surf,color,(x-r,y),(x+r,y),int(thickness))
	pygame.draw.line(surf,color,(x,y-r),(x,y+r),int(thickness))



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



x = 200
y = 150
xdot = 0
ydot = 0
ax = 0
ay = 0


x = [x,y,xdot,ydot]
t=.05		#delta time-step in sec.

A = [[1,0,t,0],
	[0,1,0,t],
	[0,0,1,0],
	[0,0,0,1]]

B = [[0.5*t**2, 0],
	 [0, 0.5*t**2],
	 [t,        0],
	 [0,        t]]

C = [[1,0,0,0],    #mesurement matrix
	 [0,1,0,0]]

u = [ax,ay]

P = [[1000,0,0,0],		# initial state covariance
	 [0,1000,0,0],
	 [0,0,1000,0],
	 [0,0,0,1000]] 	

R = [[50,0,0,0],		# model noise covariance
	 [0,50,0,0],
	 [0,0,30,0],
	 [0,0,0,30]] 		 

Q = [[200,0],		# measurement noise covariance
	 [0,200]]		

time = 0


w,h = 600,400
screen = pygame.display.set_mode((w,h))
clock = pygame.time.Clock()


x0_kalman = np.array(x)+np.array([200,0,0,0])   # initializing the kalman filter with initial guess


kalman_history = []
real_history = []
measurement_history=[]

running = True

with draw.animate_video('kalman(linear).gif', draw_frame, duration=1/15) as anim:

	while running:
		clock.tick(20)

		for ev in pygame.event.get():
			if ev.type == pygame.QUIT:
				running = False
			if ev.type == pygame.KEYDOWN:
				if ev.key == pygame.K_ESCAPE:
					running = False

		xdot = 80*cos(time)
		ydot = 80*sin(time)
		time+=t


		x_real = np.dot(A,x) + np.dot(B,u) + np.random.randn(4)
		z = mult([C,x_real[:]]) + 10*np.random.randn(2)
		x = [x_real[0],x_real[1],xdot,ydot]

		x_kalman,P = kalman(x0_kalman,A,B,C,P,R,Q,u,z)
		x0_kalman=x_kalman[:]

		kalman_history.append(x_kalman[0:2])
		real_history.append(x_real[0:2])
		measurement_history.append(z)


		screen.fill((0,0,0))
		pygame.draw.circle(screen,(255,255,0),(int(x[0]),int(x[1])),20)
		drawPlus(screen,(int(x_kalman[0]),int(x_kalman[1])+0),5,3,(0,150,255))
		drawPlus(screen,(int(x[0]),int(x[1])+0),18,2,(5,5,5))

		for i in kalman_history:
			pygame.draw.circle(screen,(0,255,255), (int(i[0]),int(i[1])), 2)

		for i in real_history:
			pygame.draw.circle(screen,(255,255,0), (int(i[0]),int(i[1])), 2)

		for i in measurement_history:
			pygame.draw.circle(screen,(255,0,0), (int(i[0]),int(i[1])), 2)

		if len(kalman_history)>130:
			kalman_history.pop(0)
			real_history.pop(0)
			measurement_history.pop(0)

		pygame.display.update()


		anim.draw_frame(kalman_history, real_history, measurement_history)

sys.exit()
