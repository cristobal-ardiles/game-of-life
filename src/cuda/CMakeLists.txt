enable_language(CUDA)
set(CUDA_SOURCES main.cu kernel.cu)
add_executable(${PROJECT_NAME}CUDA ${CUDA_SOURCES})
set_property(TARGET ${PROJECT_NAME}CUDA PROPERTY CUDA_ARCHITECTURES 50)

