import numpy as np
# import sympy as sym
# import scipy.sparse as sps
import matplotlib.pyplot as plt
# from scipy.constants import c, mu_0, epsilon_0
# import time

# lam = 1 #wavelength
# # width, N = 10*lam, 350
# # width, N = 20*lam, 301
# width, N = 20*lam, 201
# # width, N = 10*lam, 250
# (X, Y), IDX = np.meshgrid(np.linspace(0,width,N), np.linspace(0,width,N), indexing='ij'), np.arange(N*N).reshape(N,N)
# h, omega, mu, epsilon, sigma = width/N, 2*np.pi*c/lam, mu_0, epsilon_0*np.ones_like(X,dtype=complex), np.ones_like(X)
# dist = np.sqrt((X-width/2)**2+(Y-width/2)**2)
#
# #nanoantenna
# n = 2
# guide_w = (lam / n) * 0.4
# epsilon[np.logical_and(Y>width/2-guide_w/2,Y<width/2+guide_w/2)] = (n**2) * epsilon_0
# epsilon[X>width/2] = 1 * epsilon_0
# epsilon[dist < 2*lam/n] = (n**2)*epsilon_0
#
# sigma = np.maximum(10*(X-width/2)**18/(width/2)**18, 10*(Y-width/2)**18/(width/2)**18)
# J = np.zeros((N,N))#; J[N//4,N//2] = 1.0
# x, y, eps = sym.symbols('x y eps') #symbolic form
# basis, nbase = [(h-x)*(h-y)/h**2, x*(h-y)/h**2, (h-x)*y/h**2, x*y/h**2], 4 # max at [(0,0), (h,0), (0,h), (h,h)]
# pts = [(0,0), (1,0), (0,1), (1,1)] #which corner is for which basis
# #plane wave:
# # theta = np.pi+np.pi/6; kx, ky = np.cos(theta), np.sin(theta)
# # Esrc = sym.exp(1j*(2*sym.pi/lam)*(kx*x+ky*y))
# #point source
# x0, y0 = sym.symbols('x0 y0')
# Esrc = sym.hankel1(0,2*sym.pi/lam*sym.sqrt((x-x0)**2+(y-y0)**2))
# helmholtz = lambda g: omega**2*mu*eps*g + (g.diff(x).diff(x) + g.diff(y).diff(y)) #no 'by parts, so +
# Hsrc = sym.simplify(helmholtz(Esrc))
# inner = lambda f, g: complex(sym.integrate(sym.integrate(f * g, (x,0,h)), (y,0,h)).evalf())
# Kdata, JFdata, EFdata, rows, cols, start = [],[],[],[],[], time.time()
# for i,f,pt_i in zip(range(nbase),basis,pts):
#     for j,g,pt_j in zip(range(nbase),basis,pts):
#         row_slc, col_slc = np.s_[pt_i[0]:(N-1+pt_i[0]), pt_i[1]:(N-1+pt_i[1])], np.s_[pt_j[0]:(N-1+pt_j[0]), pt_j[1]:(N-1+pt_j[1])]
#         newK = (omega**2*mu*epsilon[:N-1,:N-1]*inner(f,g) - (1/(1+1j*sigma[:N-1,:N-1]))**2*(inner(f.diff(x),g.diff(x)) + inner(f.diff(y),g.diff(y))))
#         newJF = (-1j*omega*mu*inner(f,g))
#         newEF = inner(f,g)*sym.lambdify([x,y,x0,y0,eps], Hsrc)(X[col_slc],Y[col_slc],width/1.25,width/2,epsilon[:N-1,:N-1])
#         newEF[np.isnan(newEF)] = 0 #fix singularities in point source
#         newrows, newcols = IDX[row_slc], IDX[col_slc]
#         Kdata, JFdata = np.hstack((Kdata,newK.flatten())), np.hstack((JFdata,newJF*np.ones((N-1)*(N-1))))
#         EFdata = np.hstack((EFdata,newEF.flatten()))
#         rows, cols = np.hstack((rows,newrows.flatten())), np.hstack((cols,newcols.flatten()))
# K, JF, EF = sps.coo_matrix((Kdata,(rows,cols)),shape=(N*N,N*N)), sps.coo_matrix((JFdata,(rows,cols)),shape=(N*N,N*N)), sps.coo_array((EFdata,(rows,np.zeros_like(rows))), shape=(N*N,1))
# Esct, runtime = sps.linalg.spsolve(K.tocsr(), JF.tocsr() @ J.flatten()[:,None] - EF).reshape(N,N), time.time() - start
# Esrc = sym.lambdify([x,y,x0,y0],Esrc)(X,Y,width/1.25,width/2)
# Esrc[np.isnan(Esrc)] = 1 #fix singularities
# Etot = Esct + Esrc
# print(f'solved {N*N} DoF','in %3.2fs' % runtime)
# fig, ax = plt.subplots()
# ax.imshow(np.abs(Etot).T, cmap='jet', origin='lower', vmin=0, vmax=np.max(np.abs(Etot)))
# # ax.imshow(np.abs(Etot).T, cmap='jet', origin='lower', vmin=0, vmax=5)
# # plt.imshow(np.real(Etot).T, cmap='RdBu', origin='lower',vmin=-np.max(np.abs(Etot)), vmax=np.max(np.abs(Etot)))
# # plt.show()
# # exit()

from moderngl import *
import moderngl
from pyrr import Matrix44 #for perspective
import moderngl_window as mglw #window generation

class FieldPlot(mglw.WindowConfig):
    title = "Mandelbrother"
    gl_version = (3, 3)
    window_size = (1080, 1080)
    aspect_ratio = 1
    resizable = True
    fullscreen = True
    cursor = False
    shape = (200, 200)
    xl, yl = -4,-4
    xu, yu = 4,4
    #Uniforms;
    center = None
    velocity = (0,0)
    scale = None
    zoom = 0
    order = None
    sweep = 0
    cycles = None
    # mouse = (0,0) #current mouse position

    def __init__(self, **kwargs):
        # print(kwargs)
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                //#version 330
                #version 450 core
                in vec2 in_position;
                //in vec2 in_field;
                out vec2 v_vert;
                out vec2 v_field;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    v_vert = in_position;
                    //v_field = in_field;
                    v_field = in_position;
                }
            ''',
            fragment_shader='''
                //#version 330
                #version 450 core

                in vec2 v_vert;
                in vec2 v_field;
                out vec4 f_color;

                uniform vec2 center;
                uniform float scale;
                uniform int cycles;
                uniform float order;

                vec3 inferno(float x) { //this is just a fourth order fit of the inferno colormap
                    float r = 128.333*x + 1447.333*pow(x,2) - 2235.333*pow(x,3) + 906.666*pow(x,4);
                    float g = 35*x + 44.666*pow(x,2) + 496*pow(x,3) - 330.666*pow(x,4);
                    float b = 551*x - 190*pow(x,2) - 1424*pow(x,3) + 1120*pow(x,4);
                    return vec3(r/255, g/255, b/255);
                }

                vec2 csum(vec2 a, vec2 b) {
                    return vec2(a.x+b.x, a.y+b.y);
                }

                vec2 cdiff(vec2 a, vec2 b) {
                    return vec2(a.x-b.x, a.y-b.y);
                }

                vec2 cscale(float s, vec2 a) {
                    return vec2(s*a.x, s*a.y);
                }

                vec2 cprod(vec2 a, vec2 b) {
                    return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
                }

                vec2 cpow(vec2 a, float d) {
                    float mag = pow(sqrt(a.x*a.x + a.y*a.y),d);
                    float theta = d*atan(a.y,a.x);
                    return vec2(mag * cos(theta), mag * sin(theta));
                }

                vec2 conj(vec2 a) {
                    return vec2(a.x,-a.y);
                }

                void main() {
                    //calculate mandelbrot for each point
                    // int cycles = 20;
                    vec2 zn = vec2(0.0, 0.0);
                    float v = 0;
                    int count = 0;
                    for (int i = 0; i < cycles; i++) {
                        vec2 zp = zn;

                        zn = csum(cprod(zp, zp), cdiff(cscale(scale, v_field), center)); //MANDELBROT z^2 + c
                        //zn = csum(cprod(zp, cprod(zp, zp)), cdiff(cscale(scale, v_field), center)); //CUBIC z^3 + c
                        //zn = csum(cdiff(cprod(zp, zp), zp), cdiff(cscale(scale, v_field), center)); //z^2 - z + c
                        //zn = csum(csum(cprod(zp, zp), cscale(1/10.0,conj(zp))), cdiff(cscale(scale, v_field), center)); //z^2 + z' + c //THIS ONE IS SUPER COOL
                        //zn = csum(conj(cprod(zp, zp)), cdiff(cscale(scale, v_field), center)); //MANDELBROT (z^2)' + c
                        //zn = csum(conj(zp), cdiff(cscale(scale, v_field), center)); //MANDELBROT z' + c
                        //zn = csum(conj(zp), cdiff(cscale(scale, v_field), center)); //MANDELBROT z' + c

                        //zn = csum(cpow(zp, order), cdiff(cscale(scale, v_field), center)); //DYNAMIC

                        //zn = vec2(-zp2.x - zn.x + (scale * v_field.x - center.x), zp2.y + zn.y + (scale * v_field.y - center.y));
                        //zn = vec2(zp2.x - (scale * v_field.x - center.x), zp2.y - (scale * v_field.y - center.y)); //part looks like a muscle cross-section
                        //zn = vec2(2*zp2.x + (scale * v_field.x - center.x), zp2.y + (scale * v_field.y - center.y));
                        v = sqrt(zn.x*zn.x + zn.y*zn.y);
                        if (v > 2.0) {
                            break;
                        }
                        count += 1;
                    }
                    //float E_norm = sqrt(v_field[0]*v_field[0] + v_field[1]*v_field[1]);
                    //float E_norm = sqrt(zn[0]*zn[0] + zn[1]*zn[1]);
                    //f_color = vec4(inferno(E_norm), 0.0);
                    //f_color = vec4(inferno(v/10-0.5), 0.0);
                    //f_color = vec4(inferno(v), 0.0);
                    //f_color = vec4(0.0,v/5,0.0,0.0);
                    //f_color = vec4(inferno(v/5),0.0);
                    //f_color = vec4(inferno(count/(cycles/1.5)),0.0);

                    //f_color = vec4(inferno(count/(cycles/1.2)),0.0); //VERY GOOD
                    f_color = vec4(inferno(count/(cycles/1.3)),0.0); //VERY GOOD
                }
            ''',
        )

        # print(self.prog._members.keys())

        self.center = self.prog['center']
        self.scale = self.prog['scale']
        self.cycles = self.prog['cycles']
        # self.order = self.prog['order']

        self.center.value = (0.7,0)
        self.scale.value = 2
        # self.cycles.value = 500
        self.cycles.value = 300
        # self.cycles.value = 100
        # self.order.value = 2

        self.vaos = []

        #set sample coordinates
        X, Y = np.meshgrid(np.linspace(self.xl,self.xu,self.shape[0]), np.linspace(self.xl,self.xu,self.shape[0]))
        IDX = np.arange(self.shape[0]*self.shape[1]).reshape(self.shape)

        # E_max = np.max(np.abs(Etot))
        self.positions = np.array(np.hstack((2*((X.flatten()[:,None]-self.xl)/(self.xu-self.xl)-0.5),
                                        2*((Y.flatten()[:,None]-self.yl)/(self.yu-self.yl)-0.5))),np.float32)
        # fields = np.array(np.hstack((np.real(Etot.flatten())[:,None]/E_max, np.imag(Etot.flatten())[:,None]/E_max)),np.float32)

        #append center-points for smooth rendering
        hx = np.abs(X[1,0] - X[0,0])
        hy = np.abs(Y[0,1] - Y[0,0])
        X_centerpoints, Y_centerpoints = X[:self.shape[0]-1,:self.shape[1]-1] + hx/2, Y[:self.shape[0]-1,:self.shape[1]-1] + hy/2
        self.positions = np.vstack((self.positions, np.array(np.hstack((2*((X_centerpoints.flatten()[:,None]-self.xl)/(self.xu-self.xl)-0.5),
                                                              2*((Y_centerpoints.flatten()[:,None]-self.yl)/(self.yu-self.yl)-0.5))),np.float32)))

        # Eavg = (Etot[:N-1,:N-1] + Etot[:N-1,1:] + Etot[1:,:N-1] + Etot[1:,1:]) / 4
        # fields =  np.vstack((fields, np.array(np.hstack((np.real(Eavg.flatten())[:,None]/E_max, np.imag(Eavg.flatten())[:,None]/E_max)),np.float32)))

        #need four triangles per square element
        ncenters = X_centerpoints.size
        elems1 = np.hstack((IDX[:self.shape[0]-1,:self.shape[1]-1].flatten()[:,None], IDX[1:,:self.shape[1]-1].flatten()[:,None], IDX.size + np.arange(ncenters).flatten()[:,None]))
        elems2 = np.hstack((IDX[1:,:self.shape[1]-1].flatten()[:,None], IDX[1:,1:].flatten()[:,None], IDX.size + np.arange(ncenters).flatten()[:,None]))
        elems3 = np.hstack((IDX[1:,1:].flatten()[:,None], IDX[:self.shape[0]-1,1:].flatten()[:,None], IDX.size + np.arange(ncenters).flatten()[:,None]))
        elems4 = np.hstack((IDX[:self.shape[0]-1,1:].flatten()[:,None], IDX[:self.shape[0]-1,:self.shape[1]-1].flatten()[:,None], IDX.size + np.arange(ncenters).flatten()[:,None]))
        elements = np.array(np.vstack((elems1,elems2,elems3,elems4)), dtype=np.int32)

        self.vao = mglw.opengl.vao.VAO(name='fields')
        self.vao.buffer(self.positions, '2f', ['in_position'])
        # self.vao.buffer(fields, '2f', ['in_field'])
        self.vao.index_buffer(elements) #doesn't change

        self.ctx.enable(moderngl.DEPTH_TEST)


    def render(self, time, frame_time):
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        #update coordinates
        # positions = np.array(np.hstack((2*(X.flatten()[:,None]/width-0.5), 2*(Y.flatten()[:,None]/width-0.5))),np.float32)
        self.scale.value = self.scale.value * (1 + self.zoom * frame_time)
        # self.order.value = self.order.value * (1 + self.sweep * frame_time)
        speed = 0.5 * self.scale.value * frame_time
        self.center.value = (self.center.value[0] + speed*self.velocity[0], self.center.value[1] + speed*self.velocity[1])

        # print(self.center.value, self.scale.value)

        # #solve for the current point
        # rows = [int(p[0][0]*N) for p in self.sources + [(self.mouse,self.phase)]]
        # cols = [int(p[0][1]*N) for p in self.sources + [(self.mouse,self.phase)]]
        # amps = [np.exp(1j*p[1]) for p in self.sources + [(self.mouse,self.phase)]]
        # J = sps.coo_matrix((amps, (IDX[rows,cols],np.zeros_like(rows))), shape=(N*N,1), dtype=complex)
        # Etot = self.scatter.solve((self.JFcsr @ J).toarray()).reshape(N,N)
        # E_max = np.max(np.abs(Etot))

        # fields = np.array(np.hstack((np.real(Etot.flatten())[:,None]/E_max, np.imag(Etot.flatten())[:,None]/E_max)), dtype=np.float32)
        # Eavg = (Etot[:N-1,:N-1] + Etot[:N-1,1:] + Etot[1:,:N-1] + Etot[1:,1:]) / 4

        # fields =  np.vstack((fields, np.array(np.hstack((np.real(Eavg.flatten())[:,None]/E_max, np.imag(Eavg.flatten())[:,None]/E_max)),np.float32)))
        # self.vao._buffers[1].buffer.write(data=np.array(fields,dtype=np.float32).data)

        self.vao._buffers[0].buffer.write(data=np.array(self.positions,dtype=np.float32).data)
        self.vao.render(self.prog)
        # del Etot
        # del fields

    def resize(self, width: int, height: int):
        self.window_size = (width, height)


    # self.center.value = (0,0)
    # self.scale.value = 2
    # self.cycles.value = 100
    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        # Key presses
        if action == keys.ACTION_PRESS:
            if key == keys.LEFT:
                self.velocity = (1,self.velocity[1])
            if key == keys.RIGHT:
                self.velocity = (-1,self.velocity[1])
            if key == keys.UP:
                self.velocity = (self.velocity[0],-1)
            if key == keys.DOWN:
                self.velocity = (self.velocity[0],1)
            if key == keys.A:
                self.zoom = -1 
            if key == keys.Z:
                self.zoom = 1 
            # if key == keys.S:
            #     self.sweep = 1
            # if key == keys.X:
            #     self.sweep = -1 

        if action == keys.ACTION_RELEASE:
            if key == keys.LEFT:
                self.velocity = (0,self.velocity[1])
            if key == keys.RIGHT:
                self.velocity = (0,self.velocity[1])
            if key == keys.UP:
                self.velocity = (self.velocity[0],0)
            if key == keys.DOWN:
                self.velocity = (self.velocity[0],0)
            if key == keys.A:
                self.zoom = 0
            if key == keys.Z:
                self.zoom = 0
            # if key == keys.S:
            #     self.sweep = 0
            # if key == keys.X:
            #     self.sweep = 0


    # def mouse_position_event(self, x, y, dx, dy): #given in pixels on screen
    #     # print("Mouse position pos={} {} delta={} {}".format(x, y, dx, dy))
    #     self.mouse = (x/self.window_size[0], -y/self.window_size[1])

    # def mouse_press_event(self, x, y, button):
    #     # print("Mouse button {} pressed at {}, {}".format(button, x, y))
    #     # print("Mouse states:", self.wnd.mouse_states)
    #     self.sources += [(self.mouse, self.phase)]

    # def mouse_scroll_event(self, x_offset, y_offset): #y is +1 for scroll wheel going up and -1 for scroll wheel going down
    #     self.phase += 2*np.pi/36 * y_offset
    #     self.phase = self.phase % (2*np.pi)
    #     # print(self.phase)
    #     # print("mouse_scroll_event", x_offset, y_offset)

FieldPlot.run()
