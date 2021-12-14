


import mnist_loader,neural_net as network,pygame,pickle,numpy as np
from math import floor
from time import sleep


OPT = int(input(
"""[1] train a network
[2] load an existing one
"""))

if OPT == 1:
    yn = input('default options (y/n) ~> ')
    if yn == 'y' or yn == 'yes':
        hdn = [784,20,10]
        epochs = 30
        eta = 3.0
    elif yn == 'n' or yn =='no':
        hdn_size = int(input('amount of hidden layers ~> '))
        hdn = []
        for x in range(hdn_size):
            hdn.append(int(input(f'neurons for layer {x+1} ~> ')))
        hdn.insert(0,784)
        hdn.append(10)
        print(hdn)
        epochs = int(input('amount of generations/epochs ~> '))
        eta = float(input('eta/learning rate (float/int) ~> '))
    else:
        print('invalid option')
        exit()

    print('training started..')
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    net = network.Network(hdn)
    net.SGD(training_data, epochs, 10,eta, test_data=test_data)

    print('\n\n\ntraining has finished')
    yn = input('save network (y/n) ~> ')
    if yn == 'y' or yn == 'yes':
        while True:
            fn = input('file name ~> ')
            if fn.count('.') <= 0: fn+=".pkl"
            try:
                open(fn,'r')
                print(f'file \'{fn}\' already exists')
            except FileNotFoundError:
                break
        pickle.dump(net,open(fn,"wb"))
        print(f'network saved to \'{fn}\'\n')

elif OPT == 2:
    filename = input('file name ~> ')
    if filename.count('.') <= 0: filename+=".pkl"
    print(f'loading \'{filename}\'')
    net = pickle.load(open(filename,'rb'))
    print('finished loading')

else:
    print('invalid option')
    exit()
# NETWORK




# GAME


pygame.init()
screen = pygame.display.set_mode((1000, 784))

pygame.display.set_caption('Drawing AI')
#colors = numpy.random.randint(0, 255, size=(4, 3))

BLACK = (0,0,0)
WHITE = (255, 255, 255)
RED = (255,0,0)

#Make screen white
screen.fill(BLACK)

BLACK_L = [0,0,0]



SCREEN_L = []
for i in range(784):
    SCREEN_L.append(BLACK_L)

font = pygame.font.Font(pygame.font.get_default_font(), 15)
font2 = pygame.font.Font(pygame.font.get_default_font(), 70)
def updateThread():
    tmp=[]
    for i in range(784):
        tmp.append(SCREEN_L[i][0]/255)

    a=net.feedforward(np.array(tmp).reshape(784,1))
    for i in range(10):
        _s = f"{i}: {round(a[i][0]*100,2)}%"
        sum_text = font.render(_s,True,(255,255,255))
        screen.blit(sum_text,(800,200+20*(i+1)))

    max_txt = font2.render(str(np.argmax(a)),True,(255,255,255))
    screen.blit(max_txt,(800,20))




frc = 0
def updateScreen():
    c=0
    for i in range(784):
        if i != 0 and i % 28 == 0:
            c+=1
        pygame.draw.rect(screen, SCREEN_L[i], (28*(i%28), 28*c, 28*((i%28)+1), 28*(c+1)))
    pygame.draw.rect(screen, BLACK, (784, 0, 900, 784))
    pygame.draw.rect(screen, (0,0,255), (850, 734, 900, 784))
    for i in range(28):
        pygame.draw.line(screen, RED, (0, i*28), (784, i*28), 1)
        pygame.draw.line(screen, RED, (i*28, 0), (i*28, 784), 1)
    pygame.draw.line(screen, RED, (28*28, 0), (28*28, 784), 1)

updateScreen()

while True:
    for event in pygame.event.get():
        if event.type == 256: # 256 = QUIT
            pygame.quit()
            exit()
        elif event.type == 1024 or event.type == 1025: # 1024 = MOUSEMOTION, 1025 = MOUSEBUTTONDOWN
            if pygame.mouse.get_pressed(3)[0]:
                pos = pygame.mouse.get_pos()
                if pos[0] > 850 and pos[1] > 734:
                    for i in range(784):
                        SCREEN_L[i] = BLACK_L
                    updateScreen()
                elif pos[0] < 784:
                    x,y=floor(pos[0]/28),floor(pos[1]/28)
                    if SCREEN_L[28*y+x] != WHITE:
                        SCREEN_L[28*y+x] = WHITE
                        updateScreen()
                        updateThread()
            elif pygame.mouse.get_pressed(3)[2]:
                pos = pygame.mouse.get_pos()
                if pos[0] < 784:
                    x,y=floor(pos[0]/28),floor(pos[1]/28)
                    if SCREEN_L[28*y+x] != BLACK:
                        SCREEN_L[28*y+x] = BLACK
                        updateScreen()
                        updateThread()



    pygame.display.update()

