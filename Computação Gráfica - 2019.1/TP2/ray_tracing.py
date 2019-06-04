#-----------------------------------------IMPORT----------------------------------------------
import sys
import numpy as np
from math import sqrt
from math import pow
from math import cos
from math import tan

#--------------------------------------FUNÇÃO UNIT VECTOR-----------------------------------------------

#retorna um vetor de tamanho 1 na direção de v
def unit_vector(vector):
	v=vector/np.linalg.norm(vector)
	return v

#------------------------------------------FUNÇÃO SCHLICK-------------------------------------------------

def schlick(cosine, ref_idx):
	r0 = (1 - ref_idx) / (1 + ref_idx)
	r0 = r0*r0
	return r0 + (1-r0)*pow((1 - cosine), 5)

#------------------------------------------FUNÇÃO DE REFLEXÃO---------------------------------------------

def reflect(v, n):
	v=np.array(v)
	n=np.array(n)
	return v - 2* np.dot( (v),  (n)) *n

#--------------------------------------------FUNÇÃO DE REFRAÇÃO--------------------------------------------

def refract(v, n, ni_over_nt, refracted):
	n = np.array(n)
	uv = np.array(unit_vector(v))
	dt = np.dot( (uv),  (n)) 
	discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt)
	if discriminant > 0:
		refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant)
		return True, refracted
	else:
		return False, None

#--------------------------------------------FUNÇÃO RANDOM SCENE--------------------------------------------

def random_scene():
	n = 250 
	List = []
	List.append(sphere([0, -1000, 0], 1000, lambertian([0.5, 0.5, 0.5])))
	i = 1
	for cont in range (n):
		choose_mat = np.random.random()
		center = np.array([np.random.uniform(-11,11)+0.9*np.random.random(), 0.2, np.random.uniform(-11, 11)+0.9*np.random.random()])
		if (center - np.array([4, 0.2, 0])).size > 0.9:
			if choose_mat < 0.8:
				List.append(sphere(center, 0.2, lambertian([0.8*np.random.random(), 0.3*np.random.random(), 0.3*np.random.random()])))
				i += 1
			elif choose_mat < 0.95:
				List.append(sphere(center, 0.2, metal([0.5*(1 + np.random.random()), 0.5*(1 + np.random.random()), 0.5*(1 + np.random.random())], 0.5*np.random.random())))
				i += 1
			else:
				List.append(sphere(center, 0.2, dieletric(1.5)))	
				i += 1

	List.append(sphere([0, 1, 0], 1.0, dieletric(1.5)))	
	i += 1
	List.append(sphere([-4, 1, 0], 1.0, lambertian([0.258824, 0.435294, 0.258824])))
	i += 1
	List.append(sphere([4, 1, 0], 1.0, metal([0.7, 0.6, 0.5], 0.0)))
	i += 1
	back = hitable_list(List, i)
	return back


#----------------------------------------FUNÇÃO RANDOM SPHERE---------------------------------------------

def random_in_unit_sphere():
	theta = 2*np.pi*np.random.random()
	r=sqrt(np.random.random())
	p = np.array([r*np.cos(theta), r*np.sin(theta), np.random.random()])
	return p

#--------------------------------------FUNÇÃO RANDOM IN UNIT DISK----------------------------------------

def random_in_unit_disk():
	p = np.zeros(3)
	while True:
		p = 2.0*np.array([np.random.random(),np.random.random(), 0]) - np.array([1, 1, 0])
		if np.dot(p, p) < 1.0:
			break
	return p

#----------------------------------------------CLASSES-------------------------------------------------
#Classe raio
class ray:

	def __init__(self, a, b):
		self.A=a
		self.B=b

	def origin(self):
		return self.A

	def direction(self):
		return self.B

	def point_at_parameter(self, t):
		return self.A+(t*self.B)

#Classe hit_record
class hit_record:

	def __init__(self, tx, px, normalx, material):
		self.t = tx
		self.p = px
		self.normal = normalx
		self.mat_ptr = material
#Classe pai hitable
class hitable:

	def hit(self, r, t_min, t_max, material):
		pass

#Classe filho sphere
class sphere(hitable):

	def __init__(self,cen, r, material):
		self.center = cen
		self.radius = r
		self.rec= hit_record(0, [0, 0, 0], [0, 0, 0], material)
		self.mat_ptr= material

	def hit(self, r, t_min, t_max):
		oc=np.array(r.origin()) - self.center
		a=np.dot(r.direction(),  r.direction())
		b=(np.dot( (oc),  (r.direction())))
		c=(np.dot( (oc),  (oc))) - self.radius*self.radius
		discriminant = b*b - a*c
		if discriminant > 0:
			temp=(-b - sqrt(discriminant))/a
			if temp < t_max and temp > t_min:
				self.rec.t=temp
				self.rec.p=r.point_at_parameter(self.rec.t)
				self.rec.normal= (self.rec.p - self.center) / self.radius
				self.rec.mat_ptr = self.mat_ptr
				return True
			temp = (-b + sqrt(discriminant)) / a
			if temp < t_max and temp > t_min:
				self.rec.t = temp
				self.rec.p = r.point_at_parameter(self.rec.t)
				self.rec.normal = (self.rec.p - self.center) / self.radius
				self.rec.mat_ptr = self.mat_ptr
				return True
		return False

#Classe Filho hitable_list
class hitable_list(hitable):

	def __init__(self, l, n):
		self.List = l
		self.list_size = n
		self.rec= hit_record(0, [0, 0, 0], [0, 0, 0], material)
		self.mat_ptr = [0, 0, 0]

	def hit(self, r, t_min, t_max):
		hit_anything = False
		closest_so_far = t_max
		for cont in range(self.list_size):
			if self.List[cont].hit(r, t_min, closest_so_far ):
				hit_anything = True
				closest_so_far = self.List[cont].rec.t
				self.rec = self.List[cont].rec
		return hit_anything

#Classe Camera 
class camera:

	def __init__(self, lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist):
		self.lens_radius = aperture / 2
		vup = np.array(vup)
		theta = vfov*np.pi/180
		half_height = tan(theta/2)
		half_width = aspect * half_height
		self.origin = lookfrom
		self.w = unit_vector(lookfrom - lookat)
		self.u = unit_vector(np.cross(vup, self.w))
		self.v = np.cross(self.w, self.u)
		self.lower_left_corner = self.origin - half_width*focus_dist*self.u - half_height*focus_dist*self.v - focus_dist*self.w
		self.horizontal = 2*half_width*focus_dist*self.u
		self.vertical = 2*half_height*focus_dist*self.v

	def get_ray(self, s, t):
		rd = self.lens_radius*random_in_unit_disk()
		offset =self.u*rd[0] + self.v*rd[1]
		return ray(self.origin + offset, self.lower_left_corner + s*self.horizontal + t*self.vertical - self.origin - offset)


#Classe pai Material
class material:

	def scatter(self, r_in):
		pass

#Classe filha Lambertian
class lambertian(material):

	def __init__(self, a):
		self.albedo = a
		self.scattered = ([0, 0, 0], [0, 0, 0])
		self.attenuation = np.zeros(3)
	
	def scatter(self, r_in, rec):
		target = rec.p + np.array(rec.normal) + random_in_unit_sphere()
		self.scattered = ray(rec.p, target - rec.p)
		self.attenuation = self.albedo
		return True


#Classe filha Metal
class metal(material):

	def __init__(self, a, f):
		if f< 1:
			self.fuzz = f
		else:
			self.fuzz = 1
		self.albedo = a
		self.scattered = ([0, 0, 0], [0, 0, 0])
		self.attenuation = np.zeros(3)

	def scatter(self, r_in, rec):
		reflected = reflect(unit_vector(r_in.direction()), rec.normal)
		self.scattered = ray(rec.p, reflected + self.fuzz*random_in_unit_sphere())
		self.attenuation = self.albedo
		return (np.dot(self.scattered.direction(),  (rec.normal)) > 0)


#Classe filha Dieletric
class dieletric(material):

	def __init__(self, ri):
		self.ref_idx = ri
		self.scattered = ([0, 0, 0], [0, 0, 0])
		self.attenuation = np.zeros(3)

	def scatter(self, r_in, rec):
		outward_normal = np.zeros(3)
		reflected = reflect(r_in.direction(), rec.normal)
		ni_over_nt = 0.0
		self.attenuation = np.array([1.0, 1.0, 1.0])
		refracted = np.zeros(3)
		reflect_prob = 0.0
		cosine = 0.0
		if np.dot( (r_in.direction()),  (rec.normal)) > 0:
			outward_normal = -rec.normal
			ni_over_nt = self.ref_idx
			cosine = self.ref_idx * np.dot( (r_in.direction()),  (rec.normal)) / r_in.direction().size
		else:
			outward_normal = rec.normal
			ni_over_nt = 1.0 / self.ref_idx
			cosine = -np.dot( (r_in.direction()),  (rec.normal)) / r_in.direction().size

		verify, refracted = refract(r_in.direction(), outward_normal, ni_over_nt, refracted)
		if verify == True:
			reflect_prob = schlick(cosine, self.ref_idx)
		else:
			self.scattered = ray(rec.p, reflected)
			reflect_prob = 1.0
		if (np.random.random() < reflect_prob):
			self.scattered = ray(rec.p, reflected)
		else:
			self.scattered = ray(rec.p, refracted)
		return True

#----------------------------------------FUNÇÃO COLOR----------------------------------------------------

#função recebe um objeto do tipo ray
def color(r, world, depth):
	if world.hit(r, 0.001, sys.float_info.max):
		if(depth < 50 and world.rec.mat_ptr.scatter(r, world.rec)):
			return np.array(world.rec.mat_ptr.attenuation) * color(world.rec.mat_ptr.scattered, world, depth+1)
		else:
			return [0, 0, 0]
	else:
		unit_direction = unit_vector(r.direction())
		t= 0.5*(unit_direction[1]+1)
		return (1.0-t) * np.array([1.0, 1.0, 1.0]) + t*np.array([0.5, 0.7, 1.0])

#-------------------------------------------FUNÇÃO PPM-------------------------------------------------

#ppm image
def ppm(x, y):

	vec3=np.zeros(3)
	file = open("teste.ppm", "w") 
	nx=x
	ny=y
	ns = 10

	file.write("P3\n" + str(int(nx)) + " " + str(int(ny)) + "\n255\n")
	
	world = random_scene()
	
	lookfrom = np.array([13.5, 1.5, 3])
	lookat = np.array([0, 0.5, -1])
	dist_to_focus = 10.0
	aperture = 0.05
	cam = camera(lookfrom, lookat, [0, 1, 0], 20, float(nx)/float(ny), aperture, dist_to_focus) 
	
	for cont in range((ny-1),-1,-1):
		for cont2 in range(nx):
			col = np.array([0.0, 0.0, 0.0])
			for s in range (ns):
				u=float(cont2 + np.random.random())/float(nx)
				v=float(cont + np.random.random())/float(ny)
				r=cam.get_ray(u, v)
				p=r.point_at_parameter(2.0)
				col += np.array(color(r, world, 0))

			col /= float(ns)
			col = [sqrt(col[0]), sqrt(col[1]), sqrt(col[2])]
			ir=(255.99*col[0])
			ig=(255.99*col[1])
			ib=(255.99*col[2])
			file.write(str(int(ir)) +" " + str(int(ig)) + " " + str(int(ib)) + "\n")

	file.close() 

#-----------------------------------------MAIN----------------------------------------------------
def main():
	ppm(200,100)
	
if __name__ == '__main__':
	main()