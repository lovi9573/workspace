/*

Copyright (c) 2014, Dr Franck P. Vidal (franck.p.vidal@fpvidal.net),
http://www.fpvidal.net/
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the Bangor University nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/


/**
********************************************************************************
*
*	@file		welsh_dragon_glfw.cxx
*
*	@brief		Main test program.
*
*	@version	1.0
*
*	@date		11/11/2013
*
*	@author		Dr Franck P. Vidal
*
*	@section	License
*				BSD 3-Clause License.
*
*				For details on use and redistribution please refer
*				to http://opensource.org/licenses/BSD-3-Clause
*
*	@section	Copyright
*				(c) by Dr Franck P. Vidal (franck.p.vidal@fpvidal.net),
*				http://www.fpvidal.net/, Dec 2014, 2014, version 1.0,
*				BSD 3-Clause License
*
********************************************************************************
*/


//******************************************************************************
//	Include
//******************************************************************************
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
#include <GL/glew.h>
#endif

#ifdef __APPLE__
#define GL3_PROTOTYPES 1
#else
#define GL_GLEXT_PROTOTYPES 1
#endif

#define GLFW_INCLUDE_GLCOREARB 1

#include <GLFW/glfw3.h>

#include <stdlib.h>
#include <ctime>

#ifdef HAS_BUNDLE
#include <boost/filesystem.hpp>
#endif



#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"




#ifndef GVXR_TYPES_H
#include "gVirtualXRay/Types.h"
#endif

#ifndef GVXR_UNITS_H
#include "gVirtualXRay/Units.h"
#endif

#ifndef GVXR_POLYGON_MESH_H
#include "gVirtualXRay/PolygonMesh.h"
#endif

#ifndef GVXR_CUBE_MESH_H
#include "gVirtualXRay/CubeMesh.h"
#endif

#ifndef GVXR_XRAY_BEAM_H
#include "gVirtualXRay/XRayBeam.h"
#endif

#ifndef GVXR_XRAY_DETECTOR_H
#include "gVirtualXRay/XRayDetector.h"
#endif

#ifndef GVXR_OPENGL_UTILITIES_H
#include "gVirtualXRay/OpenGLUtilities.h"
#endif

#ifndef GVXR_MATRIX4X4_H
#include "gVirtualXRay/Matrix4x4.h"
#endif

#ifndef GVXR_XRAY_RENDERER_H
#include "gVirtualXRay/XRayRenderer.h"
#endif

#ifndef GVXR_SHADER_H
#include "gVirtualXRay/Shader.h"
#endif

#ifndef GVXR_UTILITIES_H
#include "gVirtualXRay/Utilities.h"
#endif

#ifndef GVXR_EXCEPTION_H
#include "gVirtualXRay/Exception.h"
#endif

#ifndef GVXR_STEREO_HELPER_H
#include "gVirtualXRay/StereoHelper.h"
#endif

#ifndef GVXR_TEXT_RENDERER_H
#include "gVirtualXRay/TextRenderer.h"
#endif

#include "display_gl2.frag.h"
#include "display_gl2.vert.h"

#include "display_gl3.frag.h"
#include "display_gl3.vert.h"


//******************************************************************************
//	Name space
//******************************************************************************
using namespace gVirtualXRay;


//******************************************************************************
//	Defines
//******************************************************************************
#define PREFIX ".."
#define OUTDIR "output/"


//******************************************************************************
//	Constant variables
//******************************************************************************
const GLfloat g_rotation_speed(2.0);

const int NONE(0);
const int SCENE(1);
const int OBJECT(2);
const int DETECTOR(3);

const double g_initial_intraocular_distance(1.5 * cm);        // Intraocular distance
const double g_initial_fovy(45);                              // Field of view along the y-axis
const double g_initial_near(5.0 * cm);                        // Near clipping plane
const double g_initial_far(500.0 * cm);                       // Far clipping plane
const double g_initial_screen_projection_plane(10000.0 * cm); // Screen projection plane


//******************************************************************************
//	Global variables
//******************************************************************************
GLsizei g_current_main_window_width(600);
GLsizei g_current_main_window_height(600);
GLsizei g_original_main_window_width(g_current_main_window_width);
GLsizei g_original_main_window_height(g_current_main_window_height);
GLFWwindow* g_p_main_window_id(0);
GLfloat g_zoom(120.0 * cm);

int g_button(-1);
int g_button_state(-1);
bool g_use_arc_ball(false);
GLint g_last_x_position(0);
GLint g_last_y_position(0);
GLint g_current_x_position(0);
GLint g_current_y_position(0);

bool g_display_beam(true);

Matrix4x4<GLfloat> g_scene_rotation_matrix;
Matrix4x4<GLfloat> g_detector_rotation_matrix;
Matrix4x4<GLfloat> g_sample_rotation_matrix;
Matrix4x4<GLfloat> g_text_2D_projection_matrix;

GLfloat g_incident_energy(80.0 * keV);
VEC2 g_detector_size(320.0 * mm, 320.0 * mm);
Vec2ui g_number_of_pixels(640, 640);
GLfloat g_resolution(g_detector_size.getX() / g_number_of_pixels.getX());

VEC3 g_source_position(   0.0, 0.0, -40.0 * cm);
VEC3 g_detector_position( 0.0, 0.0,  10.0 * cm);
VEC3 g_detector_up_vector(0.0, 1.0,   0.0);
const VEC3 g_background_colour(0.5, 0.5, 0.5);

PolygonMesh g_polygon_data;

XRayBeam g_xray_beam;
XRayDetector g_xray_detector;
XRayRenderer g_xray_renderer;
Shader g_display_shader;

StereoHelper g_stereo_helper;

bool g_is_xray_image_up_to_date(false);
int g_rotation_mode(SCENE);
bool g_display_wireframe(false);

bool g_use_lighing(true);
bool g_display_detector(true);


clock_t g_start(0);
int g_image_computed(0);
double g_number_of_seconds(0);
double g_fps(0);
bool g_use_fullscreen(true);

#ifdef HAS_FREETYPE
bool g_display_help(true);
TextRenderer g_font_rendered;
#endif

bool g_use_left_shift_key(false);
bool g_use_right_shift_key(false);


//******************************************************************************
//	Function declaration
//******************************************************************************
//void display();
//void initGLEW();
//void initGL();
//void initFreeType();
//void framebufferSizeCallback(GLFWwindow* apWindow, int aWidth, int aHeight);
//void keyCallback(GLFWwindow* apWindow, int aKey, int aScanCode, int anAction, int aModifierKey);
//void mouseButtonCallback(GLFWwindow* apWindow, int aButton, int aButtonState, int aModifierKey);
//void cursorPosCallback(GLFWwindow* apWindow, double x, double y);
//void scrollCallback(GLFWwindow* apWindow, double xoffset, double yoffset);
void errorCallback(int error, const char* description);
void quit();
//void idle();

//void computeRotation(MATRIX4& aRotationMatrix);
void loadDetector();
void loadSource();
void loadXRaySimulator();
void loadSTLFile(const std::string& aPrefix);
void updateXRayImage();
//void render();
//void draw();
//void drawMono();
//void drawStereo();
//void setCurrentEye();
//Vec3<GLfloat> getArcballVector(int x, int y);
//void displayHelp();


//-----------------------------
int main(int argc, char** argv)
//-----------------------------
{
    try
    {
	    // Set an error callback
	    glfwSetErrorCallback(errorCallback);

	    // Register the exit callback
		atexit(quit);

		// Initialize GLFW
	    if (!glfwInit())
	    {
	    	throw Exception(__FILE__, __FUNCTION__, __LINE__, "ERROR: cannot initialise GLFW.");
	    }

        // Enable OpenGL 2.1 if possible
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

	    // Enable quad-buffering if possible
	    glfwWindowHint(GLFW_STEREO, GL_TRUE);

	    // Enable anti-aliasing
	    glfwWindowHint(GLFW_SAMPLES, 4);

		// Create the window title
		std::stringstream window_title;
		window_title << "gVirtualXRay -- DragonDemo -- GLFW";


		//TODO(Jesse Lovitt): Figure out how to make a non-window GL context to use for rendering an image.
	    // Create a windowed mode window and its OpenGL context
	    g_p_main_window_id = glfwCreateWindow(g_current_main_window_width, g_current_main_window_height, window_title.str().data(), NULL, NULL);

        // Window cannot be created 
        if (!g_p_main_window_id)
	    {
            // Disable quad-buffering
            glfwWindowHint(GLFW_STEREO, GL_FALSE);

            // Create the window
	        g_p_main_window_id = glfwCreateWindow(g_current_main_window_width, g_current_main_window_height, window_title.str().data(), NULL, NULL);
        }
        
        // Window cannot be created 
        if (!g_p_main_window_id)
	    {
	        glfwTerminate();
	    	throw Exception(__FILE__, __FUNCTION__, __LINE__, "ERROR: cannot create a GLFW windowed mode window and its OpenGL context.");
	    }

	    // Make the window's context current
	    glfwMakeContextCurrent(g_p_main_window_id);

//        // Initialise GLEW
//        initGLEW();
//
// 		// Is stereo enable
//		if (g_stereo_helper.enable())
//		{
//			std::cout << "Stereo is enable." << std::endl;
//		}
//		else
//		{
//			std::cout << "Stereo is not enable." << std::endl;
//		}
//		checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//        // Check the OpenGL version
//        std::cout << "GL:\t" << glGetString(GL_VERSION) << std::endl;
//
//        // Check the GLSL version
//        std::cout << "GLSL:\t" << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

     	// Initialise OpenGL
//		initGL();
//		checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);

        // Initialise FreeType
//        initFreeType();

		// Check the current FBO
//		checkFBOErrorStatus(__FILE__, __FUNCTION__, __LINE__);

		// Set the stereo parameters
//		g_stereo_helper.setIntraocularDistance(g_initial_intraocular_distance);
//		g_stereo_helper.setFieldOfViewY(g_initial_fovy);
//		g_stereo_helper.setNear(g_initial_near);
//		g_stereo_helper.setFar(g_initial_far);
//		g_stereo_helper.setScreenProjectionPlane(g_initial_screen_projection_plane);

		// Initialize GLFW callback
//		glfwSetKeyCallback(g_p_main_window_id, keyCallback);
//		glfwSetFramebufferSizeCallback(g_p_main_window_id, framebufferSizeCallback);
//		glfwSetMouseButtonCallback(g_p_main_window_id, mouseButtonCallback);
//		glfwSetCursorPosCallback(g_p_main_window_id, cursorPosCallback);
//		glfwSetScrollCallback(g_p_main_window_id, scrollCallback);

		// Load the data
		loadDetector();
		checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);

		loadSource();
		checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);

		loadXRaySimulator();
		checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);

		// Add the geometry to the X-ray renderer
        std::string prefix;

        prefix += PREFIX;

		prefix += "/bullet/";

		loadSTLFile(prefix);
		checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);

		g_xray_renderer.addInnerSurface(&g_polygon_data);

		// Rotate the sample
		g_sample_rotation_matrix.rotate( 90, VEC3(1, 0, 0));

		// Rotate the scene
		g_scene_rotation_matrix.rotate(-170, VEC3(0, 1, 0));

		// Update the X-ray image
		updateXRayImage();

		// Make fullscreen

		// Set the projection matrix
//		GLint width(0);
//		GLint height(0);
//		glfwGetFramebufferSize(g_p_main_window_id, &width, &height);
//		framebufferSizeCallback(g_p_main_window_id, width, height);

		// Launch the event loop
//	    while (!glfwWindowShouldClose(g_p_main_window_id))
//	    {
	    	// Render here
//	    	display();
//
//	        // Swap front and back buffers
//	        glfwSwapBuffers(g_p_main_window_id);
//
//	        // Idle callback
//	        idle();
//
//	        // Poll for and process events
//	        glfwPollEvents();
//	    }
	}
	// Catch exception if any
	catch (const std::exception& error)
	{
		std::cerr << error.what() << std::endl;
	}

	// Close the window and shut GLFW
	if (g_p_main_window_id)
	{
		glfwDestroyWindow(g_p_main_window_id);
		g_p_main_window_id = 0;
		glfwTerminate();
	}

	// Return an exit code
	return (EXIT_SUCCESS);
}


//---------
void quit()
//---------
{
	if (g_p_main_window_id)
	{
		glfwDestroyWindow(g_p_main_window_id);
        g_p_main_window_id = 0;
        glfwTerminate();
	}
}



//------------------------------------------
void loadSTLFile(const std::string& aPrefix)
//------------------------------------------
{
	std::string stl_filename("friesinbasket.stl");

	// Open the file
	FILE* p_file_descriptor(fopen(stl_filename.data(), "rb"));

	// The file is not in the same directory as the executable
	if (!p_file_descriptor)
	{
		// Use the prefix
		stl_filename = aPrefix + stl_filename;
	}
	else
	{
		fclose(p_file_descriptor);
	}

   	// Set geometry
	g_polygon_data.setFilename(stl_filename.data());
	g_polygon_data.loadSTLFile(true, true, true, true, mm, GL_STATIC_DRAW);
	g_polygon_data.mergeVertices(true);
	g_polygon_data.setHounsfieldValue(500.0);

	// The X-ray image is not up-to-date
	g_is_xray_image_up_to_date = false;
}


//-----------------
void loadDetector()
//-----------------
{
	g_xray_detector.setResolutionInUnitsOfLengthPerPixel(g_resolution);
	g_xray_detector.setNumberOfPixels(g_number_of_pixels);
	g_xray_detector.setDetectorPosition(g_detector_position);
	g_xray_detector.setUpVector(g_detector_up_vector);

	// The X-ray image is not up-to-date
	g_is_xray_image_up_to_date = false;
}


//---------------
void loadSource()
//---------------
{
	// Set the energy
	g_xray_beam.initialise(g_incident_energy);

	// Set the source position
	g_xray_detector.setXrayPointSource(g_source_position);
	//g_xray_detector.setParallelBeam();
	g_xray_detector.setPointSource();

	// The X-ray image is not up-to-date
	g_is_xray_image_up_to_date = false;
}


//----------------------
void loadXRaySimulator()
//----------------------
{
	// Initialise the X-ray renderer
    g_xray_renderer.initialise(XRayRenderer::OPENGL,
            GL_RGB16F,
            &g_xray_detector,
            &g_xray_beam);

	g_xray_renderer.useNegativeFilteringFlag(!g_xray_renderer.getNegativeFilteringFlag());

	// The X-ray image is not up-to-date
	g_is_xray_image_up_to_date = false;
}


//--------------------
void writeImage(XRayRenderer::PixelType* im, const char* fname)
//--------------------
{
	typedef itk::Image<XRayRenderer::PixelType, 2> InputImageType;

	typedef  unsigned char OutputPixelType;
	typedef itk::Image<OutputPixelType, 2> OutputImageType;

	//Get buffer into itk Image
	InputImageType::Pointer image = InputImageType::New();
	InputImageType::IndexType start;
	start[0] = 0;
	start[1] = 0;
	InputImageType::SizeType size;
	size[0] = g_xray_detector.getNumberOfPixels().getX();
	size[1] = g_xray_detector.getNumberOfPixels().getY();
	InputImageType::RegionType region;
	region.SetSize( size );
	region.SetIndex( start );
	image->SetRegions(region);
	image->Allocate();
	itk::ImageRegionIterator<InputImageType> it(image, region);
	it.GoToBegin();
	while( !it.IsAtEnd()){
		it.Set( *im);
		++it;
		++im;
	}


	typedef itk::RescaleIntensityImageFilter< InputImageType, InputImageType> RescaleType;
	RescaleType::Pointer rescale = RescaleType::New();
	rescale->SetInput(image);
	rescale->SetOutputMinimum(0);
	rescale->SetOutputMaximum(itk::NumericTraits<OutputPixelType>::max());

	typedef itk::CastImageFilter<  InputImageType, OutputImageType > FilterType;
	FilterType::Pointer filter = FilterType::New();
	filter->SetInput( rescale->GetOutput() );

	typedef itk::ImageFileWriter< OutputImageType > WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName( fname );
	writer->SetInput( filter->GetOutput() );

	try
	{
		writer->Update();
	}
	catch( itk::ExceptionObject & e )
	{
		std::cerr << "Error: " << e << std::endl;
	}
}

//--------------------
void updateXRayImage()
//--------------------
{
	// The X-ray image is not up-to-date
	if (!g_is_xray_image_up_to_date)
	{
		// Compute the X-Ray image
		g_xray_renderer.computeImage(g_sample_rotation_matrix);

		std::string ext = ".png";
		// Normalise the X-ray image
		g_xray_renderer.normalise();

		writeImage( g_xray_renderer.getXRayImage(), (OUTDIR"printXRayImage"+ext).c_str());
//		g_xray_renderer.printLBuffer(OUTDIR"printLBuffer"+ext);
//		g_xray_renderer.printSumMuxDx(OUTDIR"printSumMuxDx"+ext);
//		g_xray_renderer.printEnergyFluence(OUTDIR"printEnergyFluence"+ext);
//		g_xray_renderer.printXRayImage (OUTDIR"printXRayImage"+ext);

		// The X-ray image is up-to-date
		g_is_xray_image_up_to_date = true;
	}
}

//----------------------------------------------------
void errorCallback(int error, const char* description)
//----------------------------------------------------
{
	std::cerr << "GLFW error: " << description << std::endl;
}

////------------------------------------------
//Vec3<GLfloat> getArcballVector(int x, int y)
////------------------------------------------
//{
//	Vec3<GLfloat> P(2.0 * float(x) / float(g_current_main_window_width) - 1.0,
//			2.0 * float(y) / float(g_current_main_window_height) - 1.0,
//			0);
//
//	P.setY(-P.getY());
//
//	float OP_squared = P.getX() * P.getX() + P.getY() * P.getY();
//	if (OP_squared <= 1.0)
//	{
//		P.setZ(sqrt(1.0 - OP_squared));  // Pythagore
//	}
//	else
//	{
//		P.normalise();  // nearest point
//	}
//
//	return (P);
//}


////--------------------------------
//float radian2degree(float anAngle)
////--------------------------------
//{
//	return (180.0 * anAngle / gVirtualXRay::PI);
//}


////--------------------------------------------
//void computeRotation(MATRIX4& aRotationMatrix)
////--------------------------------------------
//{
//	if (g_use_arc_ball)
//	{
//		if (g_current_x_position != g_last_x_position || g_current_y_position != g_last_y_position)
//		{
//			Vec3<GLfloat> va(getArcballVector(g_last_x_position,    g_last_y_position));
//			Vec3<GLfloat> vb(getArcballVector(g_current_x_position, g_current_y_position));
//
//#if (defined(_WIN32) || defined(_WIN64)) && !defined(__GNUC__)
//			float angle(g_rotation_speed * radian2degree(acos(std::min(1.0, va.dotProduct(vb)))));
//#else
//			float angle(g_rotation_speed * radian2degree(acos(std::min(1.0, va.dotProduct(vb)))));
//#endif
//
//			Vec3<GLfloat> axis_in_camera_coord(va ^ vb);
//			//axis_in_camera_coord.normalize();
//
//			Matrix4x4<GLfloat> camera2object(aRotationMatrix.getInverse());
//			Vec3<GLfloat> axis_in_object_coord = camera2object * axis_in_camera_coord;
//			//axis_in_object_coord.normalize();
//
//			Matrix4x4<GLfloat> rotation_matrix;
//			rotation_matrix.rotate(angle, axis_in_object_coord);
//			aRotationMatrix = aRotationMatrix * rotation_matrix;
//
//			g_last_x_position = g_current_x_position;
//			g_last_y_position = g_current_y_position;
//		}
//	}
//}



////-------------
//void initGLEW()
////-------------
//{
//#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
//	GLenum err = glewInit();
//	if (GLEW_OK != err)
//	{
//		std::stringstream error_message;
//		error_message << "ERROR: cannot initialise GLEW:\t" << glewGetErrorString(err);
//
//        throw Exception(__FILE__, __FUNCTION__, __LINE__, error_message.str());
//	}
//#endif
//}

//
////-----------
//void initGL()
////-----------
//{
//	// Enable the Z-buffer
//	glEnable(GL_DEPTH_TEST);
//	checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//	// Set the background colour
//    glClearColor(g_background_colour.getX(), g_background_colour.getY(), g_background_colour.getZ(), 1.0);
//	checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//	glEnable(GL_MULTISAMPLE);
//	checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//	// Initialise the shaders
//	char* p_vertex_shader(0);
//	char* p_fragment_shader(0);
//
//	int z_lib_return_code_vertex(0);
//	int z_lib_return_code_fragment(0);
//
//	std::string vertex_shader;
//	std::string fragment_shader;
//
//    // Display shader
//    if (useOpenGL3_2OrAbove())
//    {
//        z_lib_return_code_vertex   = inflate(g_display_gl3_vert, sizeof(g_display_gl3_vert),   &p_vertex_shader);
//        z_lib_return_code_fragment = inflate(g_display_gl3_frag, sizeof(g_display_gl3_frag), &p_fragment_shader);
//    }
//    else
//    {
//        z_lib_return_code_vertex   = inflate(g_display_gl2_vert, sizeof(g_display_gl2_vert),   &p_vertex_shader);
//        z_lib_return_code_fragment = inflate(g_display_gl2_frag, sizeof(g_display_gl2_frag), &p_fragment_shader);
//    }
//
//	vertex_shader   = p_vertex_shader;
//	fragment_shader = p_fragment_shader;
//	delete [] p_vertex_shader;     p_vertex_shader = 0;
//	delete [] p_fragment_shader; p_fragment_shader = 0;
//
//	if (z_lib_return_code_vertex <= 0 || z_lib_return_code_fragment <= 0 || !vertex_shader.size() || !fragment_shader.size())
//	{
//		throw Exception(__FILE__, __FUNCTION__, __LINE__, "Cannot decode the shader using ZLib.");
//	}
//
//	g_display_shader.setLabels("display.vert", "display.frag");
//	g_display_shader.loadSource(vertex_shader, fragment_shader);
//	checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//	// Enable the shader
//	g_display_shader.enable();
//	GLint shader_id(g_display_shader.getProgramHandle());
//
//	GLfloat light_global_ambient[] = { 0.2, 0.2, 0.2, 1.0 };
//	GLfloat light_ambient[] = { 0.2, 0.2, 0.2, 1.0 };
//	GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
//	GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
//
//	// Handle for shader variables
//	GLuint handle(0);
//
//	handle = glGetUniformLocation(shader_id, "light_global_ambient");
//	glUniform4fv(handle, 1, &light_global_ambient[0]);
//	checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//	handle = glGetUniformLocation(shader_id, "light_ambient");
//	glUniform4fv(handle, 1, &light_ambient[0]);
//	checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//	handle = glGetUniformLocation(shader_id, "light_diffuse");
//	glUniform4fv(handle, 1, &light_diffuse[0]);
//	checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//	handle = glGetUniformLocation(shader_id, "light_specular");
//	glUniform4fv(handle, 1, &light_specular[0]);
//	checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//	VEC3 light_position(0, 0, g_zoom);
//	handle = glGetUniformLocation(shader_id, "light_position");
//	glUniform4f(handle, light_position.getX(), light_position.getY(), light_position.getZ(), 1.0);
//	checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//	// Disable the shader
//	g_display_shader.disable();
//}


////-----------------
//void initFreeType()
////-----------------
//{
//#ifdef HAS_FREETYPE
//    g_font_rendered.initialise(PREFIX"/data/arial.ttf", 16*27);
//    g_display_help = true;
//#endif
//}

//
////-----------
//void render()
////-----------
//{
//	try
//	{
//        // Enable back face culling
//        pushEnableDisableState(GL_CULL_FACE);
//        glEnable(GL_CULL_FACE);
//        glCullFace(GL_BACK);
//
//        // Enable the shader
//        pushShaderProgram();
//        g_display_shader.enable();
//        GLint shader_id(g_display_shader.getProgramHandle());
//
//        // Check the status of OpenGL and of the current FBO
//        checkFBOErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//        checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//        // Store the current transformation matrix
//        pushModelViewMatrix();
//
//        // Rotate the sample
//        g_current_modelview_matrix *= g_scene_rotation_matrix;
//
//        GLuint handle(0);
//        handle = glGetUniformLocation(shader_id, "g_projection_matrix");
//        glUniformMatrix4fv(handle, 1, GL_FALSE, g_current_projection_matrix.get());
//        checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//        handle = glGetUniformLocation(shader_id, "g_modelview_matrix");
//        glUniformMatrix4fv(handle, 1, GL_FALSE, g_current_modelview_matrix.get());
//        checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//        MATRIX4 normal_matrix(g_current_modelview_matrix);
//        handle = glGetUniformLocation(shader_id, "g_normal_matrix");
//        glUniformMatrix3fv(handle, 1, GL_FALSE, normal_matrix.get3x3());
//        checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//        GLint lighting;
//        if (g_use_lighing)
//        {
//            lighting = 1;
//        }
//        else
//        {
//            lighting = 0;
//        }
//        handle = glGetUniformLocation(shader_id,"g_use_lighting");
//        glUniform1iv(handle, 1, &lighting);
//        checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//        // Set the colour of the sample
//        {
//            const GLfloat material_ambient[]  = {0.19225,  0.0, 0.0, 1.0};
//            const GLfloat material_diffuse[]  = {0.50754,  0.0, 0.0, 1.0};
//            const GLfloat material_specular[] = {0.508273, 0.0, 0.0, 1.0};
//            const GLfloat material_shininess = 50.2;
//
//            handle = glGetUniformLocation(shader_id, "material_ambient");
//            glUniform4fv(handle, 1, &material_ambient[0]);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//            handle = glGetUniformLocation(shader_id, "material_diffuse");
//            glUniform4fv(handle, 1, &material_diffuse[0]);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//            handle = glGetUniformLocation(shader_id, "material_specular");
//            glUniform4fv(handle, 1, &material_specular[0]);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//            handle = glGetUniformLocation(shader_id, "material_shininess");
//            glUniform1fv(handle, 1, &material_shininess);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//        }
//
//        // Store the current transformation matrix
//        pushModelViewMatrix();
//
//        g_current_modelview_matrix *= g_sample_rotation_matrix;
//        applyModelViewMatrix();
//
//        // Display the sample
//        if (g_display_wireframe)
//        {
//            g_polygon_data.displayWireFrame();
//        }
//        else
//        {
//            g_polygon_data.display();
//        }
//
//        // Restore the current transformation matrix
//        popModelViewMatrix();
//
//        // Set the colour of the source
//        {
//            const GLfloat material_ambient[]  = {0.0, 0.39225,  0.39225,  1.0};
//            const GLfloat material_diffuse[]  = {0.0, 0.70754,  0.70754,  1.0};
//            const GLfloat material_specular[] = {0.0, 0.708273, 0.708273, 1.0};
//            const GLfloat material_shininess = 50.2;
//
//            handle = glGetUniformLocation(shader_id, "material_ambient");
//            glUniform4fv(handle, 1, &material_ambient[0]);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//            handle = glGetUniformLocation(shader_id, "material_diffuse");
//            glUniform4fv(handle, 1, &material_diffuse[0]);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//            handle = glGetUniformLocation(shader_id, "material_specular");
//            glUniform4fv(handle, 1, &material_specular[0]);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//            handle = glGetUniformLocation(shader_id, "material_shininess");
//            glUniform1fv(handle, 1, &material_shininess);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//        }
//
//        // Display the source
//        g_xray_detector.displaySource();
//
//        // Disable back face culling
//        glDisable(GL_CULL_FACE);
//
//        // Display the X-Ray image
//        if (g_display_detector)
//        {
//            g_xray_renderer.display();
//        }
//
//        // Display the beam
//        if (g_display_beam)
//        {
//            const GLfloat material_ambient[]  = {0.75, 0, 0.5, 0.3};
//            handle = glGetUniformLocation(shader_id, "material_ambient");
//            glUniform4fv(handle, 1, &material_ambient[0]);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//            lighting = 0;
//            handle = glGetUniformLocation(shader_id,"g_use_lighting");
//            glUniform1iv(handle, 1, &lighting);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//
//            g_xray_detector.displayBeam();
//        }
//
//#ifdef HAS_FREETYPE
//        if (g_display_help)
//        {
//            displayHelp();
//
//            // Check the status of OpenGL and of the current FBO
//            checkFBOErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//            checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//        }
//#endif
//
//        // Disable the shader
//        popShaderProgram();
//
//        // Restore the current transformation matrix
//        popModelViewMatrix();
//
//        // Restore the attributes
//        popEnableDisableState();
//
//        // Check the status of OpenGL and of the current FBO
//        checkFBOErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//        checkOpenGLErrorStatus(__FILE__, __FUNCTION__, __LINE__);
//	}
//	// Catch exception if any
//	catch (const std::exception& error)
//	{
//		std::cerr << error.what() << std::endl;
//	}
//}

//
////---------
//void draw()
////---------
//{
//	// Get the current draw buffer if needed
//	GLint back_buffer(0);
//	glGetIntegerv(GL_DRAW_BUFFER, &back_buffer);
//
//	g_stereo_helper.swapEye();
//
//	// Clear the buffers
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//	// Display the 3D scene
//	render();
//
//	// Make sure all the OpenGL code is done
//	glFinish();
//
//	// Restore the draw buffer if needed
//	glDrawBuffer(back_buffer);
//}
//

////---------------
//void drawStereo()
////---------------
//{
//	if (g_stereo_helper.enable())
//	{
//		draw();
//		draw();
//	}
//}
//

////-------------
//void drawMono()
////-------------
//{
//	draw();
//}
//

////------------
//void display()
////------------
//{
//	if (!g_image_computed)
//	{
//		g_start = clock();
//	}
//
//	// Store the current transformation matrix
//	pushModelViewMatrix();
//	pushProjectionMatrix();
//
//	// Update the X-ray image
//	updateXRayImage();
//
//    // Use stereo
//	if (g_stereo_helper.isActive())
//	{
//		drawStereo();
//	}
//    // Cannot use stereo
//	else
//	{
//		drawMono();
//	}
//
//	// Restore the current transformation matrix
//	popModelViewMatrix();
//	popProjectionMatrix();
//
//	clock_t end(clock());
//	clock_t duration(end - g_start);
//
//	g_number_of_seconds = double(duration) / CLOCKS_PER_SEC;
//
//	g_fps = ++g_image_computed / g_number_of_seconds;
//}


////-------------------------------------------------------------------------------
//void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
////-------------------------------------------------------------------------------
//{
//	if (action == GLFW_PRESS)
//	{
//		switch(key)
//		{
//#ifdef HAS_FREETYPE
//        case GLFW_KEY_H:
//            g_display_help = !g_display_help;
//            break;
//#endif
//
//		case GLFW_KEY_Q:
//		case GLFW_KEY_ESCAPE:
//			glfwSetWindowShouldClose(g_p_main_window_id, GL_TRUE);
//			break;
//
//		case GLFW_KEY_S:
//			if (g_stereo_helper.isActive())
//			{
//				g_stereo_helper.disable();
//			}
//			else
//			{
//				g_stereo_helper.enable();
//			}
//			break;
//
//
//        case GLFW_KEY_O:
//            g_stereo_helper.setIntraocularDistance(g_stereo_helper.getIntraocularDistance() + 5.0 * mm);
//            break;
//
//        case GLFW_KEY_P:
//            g_stereo_helper.setIntraocularDistance(g_stereo_helper.getIntraocularDistance() - 5.0 * mm);
//            break;
//
//        case GLFW_KEY_RIGHT:
//            {
//                VEC3 translation(g_detector_position - g_source_position);
//                translation.normalise();
//                g_source_position -= translation * 1.5 * mm;
//
//                // Set the source position
//                g_xray_detector.setXrayPointSource(g_source_position);
//
//                // The X-ray image is not up-to-date
//                g_is_xray_image_up_to_date = false;
//            }
//            break;
//
//        case GLFW_KEY_LEFT:
//            {
//                VEC3 translation(g_detector_position - g_source_position);
//                translation.normalise();
//                g_source_position += translation * 1.5 * mm;
//
//                // Set the source position
//                g_xray_detector.setXrayPointSource(g_source_position);
//
//                // The X-ray image is not up-to-date
//                g_is_xray_image_up_to_date = false;
//            }
//            break;
//
//        case GLFW_KEY_UP:
//            {
//                VEC3 translation(g_detector_up_vector);
//                translation.normalise();
//                g_source_position += translation * 1.5 * mm;
//
//                // Set the source position
//                g_xray_detector.setXrayPointSource(g_source_position);
//
//                // The X-ray image is not up-to-date
//                g_is_xray_image_up_to_date = false;
//            }
//            break;
//
//        case GLFW_KEY_DOWN:
//            {
//                VEC3 translation(g_detector_up_vector);
//                translation.normalise();
//                g_source_position -= translation * 1.5 * mm;
//
//                // Set the source position
//                g_xray_detector.setXrayPointSource(g_source_position);
//
//                // The X-ray image is not up-to-date
//                g_is_xray_image_up_to_date = false;
//            }
//            break;
//
//        case GLFW_KEY_B:
//            g_display_beam = !g_display_beam;
//            break;
//
//
//        case GLFW_KEY_W:
//            g_display_wireframe = !g_display_wireframe;
//            break;
//
//        case GLFW_KEY_N:
//            g_xray_renderer.useNegativeFilteringFlag(!g_xray_renderer.getNegativeFilteringFlag());
//            break;
//
//        case GLFW_KEY_L:
//            g_use_lighing = !g_use_lighing;
//            break;
//
//        case GLFW_KEY_D:
//            g_display_detector = !g_display_detector;
//            break;
//
//        case GLFW_KEY_1:
//			// This is a point source
//			if (g_xray_detector.getSourceShape() != XRayDetector::PARALLEL)
//			{
//				// Use a parallel source
//				g_xray_detector.setParallelBeam();
//			}
//			// This is a cubic source
//			else
//			{
//				g_xray_detector.setPointSource();
//			}
//
//			// The X-ray image is not up-to-date
//			g_is_xray_image_up_to_date = false;
//			break;
//
//		case GLFW_KEY_2:
//			// This is not a line source
//			if (g_xray_detector.getSourceShape() != XRayDetector::LINE)
//			{
//				g_xray_detector.setLineSource(g_xray_detector.getXraySourceCentre(), VEC3(1, 0, 0), 16, 10.0*mm);
//			}
//			// This is a cubic source
//			else
//			{
//				g_xray_detector.setPointSource();
//			}
//
//			// The X-ray image is not up-to-date
//			g_is_xray_image_up_to_date = false;
//			break;
//
//		case GLFW_KEY_3:
//			// This is not a square source
//			if (g_xray_detector.getSourceShape() != XRayDetector::SQUARE)
//			{
//				g_xray_detector.setSquareSource(g_xray_detector.getXraySourceCentre(), 10, 10.0*mm);
//			}
//			// This is a cubic source
//			else
//			{
//				g_xray_detector.setPointSource();
//			}
//			// The X-ray image is not up-to-date
//			g_is_xray_image_up_to_date = false;
//
//			break;
//
//		case GLFW_KEY_4:
//			// This is not a cubic source
//			if (g_xray_detector.getSourceShape() != XRayDetector::CUBE)
//			{
//				g_xray_detector.setCubicSource(g_xray_detector.getXraySourceCentre(), 5, 10.0*mm);
//			}
//			// This is a cubic source
//			else
//			{
//				g_xray_detector.setPointSource();
//			}
//
//			// The X-ray image is not up-to-date
//			g_is_xray_image_up_to_date = false;
//			break;
//
//        case GLFW_KEY_KP_ADD:
//            g_zoom -= 1.0 * cm;
//            framebufferSizeCallback(g_p_main_window_id, g_current_main_window_width, g_current_main_window_height);
//            break;
//
//        case GLFW_KEY_EQUAL:
//
//            if (g_use_left_shift_key || g_use_right_shift_key)
//            {
//                g_zoom -= 1.0 * cm;
//                framebufferSizeCallback(g_p_main_window_id, g_current_main_window_width, g_current_main_window_height);
//            }
//            break;
//
//        case GLFW_KEY_MINUS:
//        case GLFW_KEY_KP_SUBTRACT:
//            g_zoom += 1.0 * cm;
//            framebufferSizeCallback(g_p_main_window_id, g_current_main_window_width, g_current_main_window_height);
//            break;
//
//        case GLFW_KEY_LEFT_SHIFT:
//            g_use_left_shift_key = true;
//            break;
//
//        case GLFW_KEY_RIGHT_SHIFT:
//            g_use_right_shift_key = true;
//            break;
//
//        default:
//            break;
//        }
//    }
//    else if (action == GLFW_RELEASE)
//    {
//        switch (key)
//        {
//        case GLFW_KEY_LEFT_SHIFT:
//            g_use_left_shift_key = false;
//            break;
//
//        case GLFW_KEY_RIGHT_SHIFT:
//            g_use_right_shift_key = false;
//            break;
//        }
//    }
//}


////-----------------------------------------------------------------------
//void framebufferSizeCallback(GLFWwindow* apWindow, int width, int height)
////-----------------------------------------------------------------------
//{
//	if (height == 0)
//	{
//		// Prevent divide by 0
//		height = 1;
//	}
//
//	int x(0), y(0), w(width), h(height);
//
//	g_current_main_window_width = width;
//	g_current_main_window_height = height;
//
//	double screen_aspect_ratio(double(g_current_main_window_width) / double(g_current_main_window_height));
//	g_stereo_helper.setScreenAspectRatio(screen_aspect_ratio);
//
//	glViewport(x, y, w, h);
//
//	loadPerspectiveProjectionMatrix(g_initial_fovy, screen_aspect_ratio, g_initial_near, g_initial_far);
//
//	loadLookAtModelViewMatrix(0.0, 0.0, g_zoom,
//			0.0, 0.0, 0.0,
//			0.0, 1.0, 0.0);
//
//    g_text_2D_projection_matrix = buildOrthoProjectionMatrix(0,
//            g_current_main_window_width,
//            g_current_main_window_height,
//            0,
//            -1.0,
//            1.0);
//}


////--------------------------------------------
//void mouseButtonCallback(GLFWwindow* apWindow,
//						 int aButton,
//						 int aButtonState,
//						 int aModifierKey)
////--------------------------------------------
//{
//	g_button = aButton;
//	g_button_state = aButtonState;
//
//	// Use the arc ball
//	if (g_button_state == GLFW_PRESS)
//	{
//		g_use_arc_ball = true;
//
//        // Select the right transformation
//        if (g_rotation_mode == NONE)
//        {
//            if (aModifierKey & GLFW_MOD_SHIFT)
//            {
//                g_rotation_mode = SCENE;
//            }
//            else if (aModifierKey & GLFW_MOD_CONTROL)
//            {
//                g_rotation_mode = OBJECT;
//            }
//            else if (aModifierKey & GLFW_MOD_ALT)
//            {
//                g_rotation_mode = DETECTOR;
//            }
//            else
//            {
//                switch (g_button)
//                {
//                case GLFW_MOUSE_BUTTON_LEFT:
//                    g_rotation_mode = SCENE;
//                    break;
//
//                case GLFW_MOUSE_BUTTON_RIGHT:
//                    g_rotation_mode = OBJECT;
//                    break;
//
//                case GLFW_MOUSE_BUTTON_MIDDLE:
//                    g_rotation_mode = DETECTOR;
//                    break;
//
//                default:
//                    g_rotation_mode = SCENE;
//                    break;
//                }
//            }
//        }
//	}
//	// Stop using the arc ball
//	else
//	{
//		g_use_arc_ball = false;
//		g_rotation_mode = NONE;
//	}
//
//	double xpos(0);
//	double ypos(0);
//	glfwGetCursorPos (apWindow, &xpos, &ypos);
//
//	g_last_x_position = xpos;
//	g_last_y_position = ypos;
//
//	g_current_x_position = xpos;
//	g_current_y_position = ypos;
//}


////-----------------------------------------------------------------------
//void scrollCallback(GLFWwindow* apWindow, double xoffset, double yoffset)
////-----------------------------------------------------------------------
//{
//	// Scrolling along the Y-axis
//	if (fabs(yoffset) > EPSILON)
//	{
//		g_use_arc_ball = false;
//		g_zoom += yoffset * cm;
//		framebufferSizeCallback(apWindow, g_current_main_window_width, g_current_main_window_height);
//	}
//}


////--------------------------------------------------------------
//void cursorPosCallback(GLFWwindow* apWindow, double x, double y)
////--------------------------------------------------------------
//{
//	g_current_x_position = x;
//	g_current_y_position = y;
//
//	// Update the rotation matrix if needed
//	switch (g_rotation_mode)
//	{
//	case SCENE:
//		// Compute the rotation
//		computeRotation(g_scene_rotation_matrix);
//		break;
//
//	case OBJECT:
//		// Compute the rotation
//		computeRotation(g_sample_rotation_matrix);
//
//		// The X-ray image is not up-to-date
//		g_is_xray_image_up_to_date = false;
//		break;
//
//	case DETECTOR:
//		// Compute the rotation
//		computeRotation(g_detector_rotation_matrix);
//		g_xray_detector.setRotationMatrix(g_detector_rotation_matrix);
//
//		// The X-ray image is not up-to-date
//		g_is_xray_image_up_to_date = false;
//		break;
//
//	default:
//		break;
//	}
//}




////---------
//void idle()
////---------
//{
//	if ((g_image_computed % 50) == 0)
//	{
//		std::cout << "FPS:\t" << g_fps << std::endl;
//		g_image_computed = 0;
//	}
//}


////----------------
//void displayHelp()
////----------------
//{
//#ifdef HAS_FREETYPE
//    GLfloat p_background_colour [] = {
//            g_background_colour.getX(),
//            g_background_colour.getY(),
//            g_background_colour.getZ(),
//            0.0
//    };
//
//    GLfloat p_font_colour [] = {
//            1.0,
//            1.0,
//            0.0,
//            1.0
//    };
//
//    pushEnableDisableState(GL_BLEND);
//    glEnable (GL_BLEND);
//    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//
//    int x_position(15);
//    int offset(29);
//    g_font_rendered.renderText(std::string("Help"),
//            20,   x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'h\': display|hide help message"),
//            40, x_position += offset,
//                g_text_2D_projection_matrix,
//                p_background_colour,
//                p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'s\': toggle stereo"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'o|p\': adjust inter-ocular distance"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("Arrows: move the source"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'n\': toggle negative image"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'b\': display|hide X-ray beam"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'w\': wireframe|solid rendering"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'l\': enable|disable lighing"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'d\': display|hide detector"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'1\': use point source|parallel beam"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'2\': use point|line source"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'3\': use point|square source"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'4\': use point|cubic source"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'+|-\': zoom in|out"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//
//    g_font_rendered.renderText(std::string("\'shift\' + mouse, scene"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'ctrl\' + mouse, move object"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'alt\' + mouse, move detector"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    g_font_rendered.renderText(std::string("\'q\', 'Esc': quit"),
//            40, x_position += offset,
//            g_text_2D_projection_matrix,
//            p_background_colour,
//            p_font_colour);
//
//    popEnableDisableState();
//#endif
//}
