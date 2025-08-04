#include "image.cuh"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#include <chrono>

#include "filter.cuh"

int window();
void renderUI();
GLuint textureFromImage(Image image);


// Create a checkerboard texture for demo
GLuint image_texture;
// Variables for sliders
static float slider1 = 0.5f;
static float slider2 = 0.5f;
static float slider3 = 0.5f;

int display_w, display_h;

int window()
{
    // Init GLFW
    if (!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui Example", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    image_texture = textureFromImage(Image("sample/cornell/32/Render.png"));

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        glfwGetFramebufferSize(window, &display_w, &display_h);
       
        renderUI();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteTextures(1, &image_texture);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}



void renderUI(){
    ImGui::NewFrame();

    // Main window full screen (no decoration)
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize({display_w, display_h});
    ImGui::Begin("Main Window", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    // Left menu panel (fixed width)
    ImGui::BeginChild("Left Menu", ImVec2(250, 0), true);
    ImGui::Text("Sliders Menu");
    ImGui::Separator();

    ImGui::SliderFloat("Sigma Space", &slider1, 0.0f, 10.0f);
    ImGui::SliderFloat("Sigma Color", &slider2, 0.0f, 10.0f);
    ImGui::SliderFloat("Sigma Gaussian", &slider3, 0.0f, 10.0f);

    
    Image render("render/cornell/render_1.png");
    int2 shape = {render.shape.x, render.shape.y};
    std::vector<float3> render_f = fVecFromImage(render);
    std::vector<float3> out_f(totalSize(shape));

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    waveletfilterGPU(render_f.data(), out_f.data(),
        nullptr,
        nullptr,
        shape,
        5, 4, slider1, slider2, 1, slider3);

    auto end = high_resolution_clock::now();

    // Calculate the duration
    duration<double, std::milli> exec_time = end - start;
    

    ImGui::Text("Time: %.3f", exec_time.count());

    ImGui::EndChild();

    ImGui::SameLine();

    // Right panel for image (fills remaining space)
    ImGui::BeginChild("Right Panel", ImVec2(0, 0), true);
    ImGui::Text("Image Display");


    Image output(out_f.data(), shape);
    auto txt = textureFromImage(output);

    // Display the texture, scale it to fit within the panel but keep aspect ratio
    ImVec2 avail = ImGui::GetContentRegionAvail();
    float aspect = 1.0f; // checkerboard is square
    ImVec2 image_size = ImVec2(avail.x, avail.x / aspect);
    if (image_size.y > avail.y)
        image_size = ImVec2(avail.y * aspect, avail.y);

    ImGui::Image((ImTextureID)(intptr_t)txt, image_size);

    ImGui::EndChild();

    ImGui::End();

    // Rendering
    ImGui::Render();

}


GLuint textureFromImage(Image image){
    GLuint tex_id;
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape.x, image.shape.y, 0, GL_RGB, GL_UNSIGNED_BYTE, image.buffer);

    return tex_id;
}