# Pseudocódigo

### Jacobi por bloques
Esta aproximación busca obtener un buen rendimiento para calcular un resultado
aproximado por medio de la utilización de múltiples hilos en GPU y un único
hilo en CPU.

```
A: Apuntador a matriz linealizada en RAM
b: Apuntador a vector en RAM (Tiene tantos elementos como filas hay en A).
x: Apuntador a vector en RAM (Tiene tantos elementos como filas hay en A).
x_c: Apuntador a vector en RAM (Solución actual al sistema Ax=b).
x_e: Apuntador a vector en RAM (Error de cada elemento de la solución aproximada x_c).
filas: Cantidad de filas de la matriz A.
cols: Cantidad de columnas de la matriz A.

jacobi(...): Kernel para resolver el sistema en GPU.
reservar_gpu(...): Esta función recibe un entero que es el tamaño y lo multiplica por el tipo de dato para saber el tamaño exacto para reservar.
encontrar_max_err(...): Encuentra el valor máximo del vector gpu_x_e.
computar_errs(...): Halla el error de la submatriz correspondiente, utilizando gpu_x_c y gpu_x_n.

Variables globales: filas, cols, tamaño_double, rel

Funcion correr_jacobi(A, gpu_A, gpu_b, gpu_x_c, gpu_x_n,
                                     filas_gpu,  iters_para_procesar_A,
                                     iters_para_procesar_A_menos_1):
  A_ptr <- obtener_memdir(A[0])
  # recorrer la cantidad de filas que se van cargando a
  # la memoria de la GPU.(No recorre toda la matriz a la primera)
  Para j <- 0; j < iters_para_procesar_A; j <- j + 1 hacer:
    A_ptr <- A_ptr + j * filas_gpu * cols
    Si j = iters_para_procesar_A_menos_1 entonces:
      filas_gpu <- filas - j * filas_gpu
    FinSi
    copy_to_gpu(gpu_A, A_ptr, filas_gpu * cols * tamaño_double)
    jacobi(gpu_A, gpu_b, gpu_x_c, gpu_x_n, filas_gpu, cols, j * filas_gpu, rel)
    # Es posible optimizar este llamado creando un arreglo
    # de tamaño iters_para_procesar_A donde en cada posición
    # se puede almacenar j * filas_gpu y de esta manera no es necesario
    # realizar este producto en cada una de las iteraciones.
    computar_errs(
      gpu_x_c + j * filas_gpu,
      gpu_x_n + j * filas_gpu,
      gpu_x_e + j * filas_gpu,
      filas_gpu
    )
  FinPara

FinFuncion

Funcion solve:

  posiciones_A <- filas*cols

  # Reservar e inicializar en Cero
  x_c <- ceros(cols)
  total_gpu_mem <- obtener_gpu_mem()
  tamaño_double <- obtener_tamaño_double()
  # Calcular espacio requerido para almacenar cada vector
  tamaño_vector <- cols * tamaño_double
  # Calcular espacio requerido para almacenar submatriz de A.
  # Es el espacio disponible para almacenar la submatriz A
  # después de reservar espacio para los cuatro vectores y para el error.
  posiciones_disponibles <- redondear_hacia_abajo(total_gpu_mem - 4  *  tamaño_vector - tamaño_double)
  # Cantidad de filas de A que caben en la memoria de la GPU
  filas_gpu <- redondear_hacia_abajo(posiciones_disponibles/tamaño_vector)

  # Reservar en GPU.
  gpu_b <- reservar_gpu(cols)
  gpu_x_c <- reservar_gpu(cols)
  gpu_x_n <- reservar_gpu(cols)
  gpu_x_e <- reservar_gpu(cols)
  gpu_max_err <- reservar_gpu(1)
  gpu_A <- reservar_gpu(filas_gpu)

  # Copiar vectores a la GPU
  copiar_a_gpu(gpu_b, b, tamaño_vector)
  copiar_a_gpu(gpu_x_c, b, tamaño_vector)

  err <- tol + 1
  err_ptr <- obtener_memdir(err)
  cont <- 0
  # Total dividido lo que puedo tomar.
  iters_para_procesar_A <- redondear_hacia_arriba(filas/filas_gpu)
  iters_para_procesar_A_menos_1 <- iters_para_procesar_A - 1

  Mientras err > tol and cont < niter  hacer:
    Si cont % 2 = 0 entonces:
      correr_jacobi(A, gpu_A, gpu_b, gpu_x_c, gpu_x_n,
                    filas_gpu,  iters_para_procesar_A,
                    iters_para_procesar_A_menos_1)
    Sino:
      correr_jacobi(A, gpu_A, gpu_b, gpu_x_n, gpu_x_c,
                    filas_gpu,  iters_para_procesar_A,
                    iters_para_procesar_A_menos_1)
    FinSi
    encontrar_max_err(gpu_x_e, gpu_max_err)
    copiar_desde_gpu(err_ptr, gpu_max_err)
  FinMientras

FinFuncion
```
