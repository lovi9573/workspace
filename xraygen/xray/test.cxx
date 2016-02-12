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




//******************************************************************************
//	Name space
//******************************************************************************
using namespace gVirtualXRay;


//******************************************************************************
//	Defines
//******************************************************************************
#define PREFIX ".."


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
XRayBeam g_xray_beam;
XRayDetector g_xray_detector;
XRayRenderer g_xray_renderer;

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
//void errorCallback(int error, const char* description);
//void quit();
//void idle();
//
//void computeRotation(MATRIX4& aRotationMatrix);
void loadDetector();
void loadSource();
void loadXRaySimulator();
void loadSTLFile(const std::string& aPrefix);
//void updateXRayImage();
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


		// Load the data
		loadDetector();

		loadSource();

		loadXRaySimulator();

		// Add the geometry to the X-ray renderer
        std::string prefix;


		prefix += "../bullet/";

		loadSTLFile(prefix);

		g_xray_renderer.addInnerSurface(&g_polygon_data);
//
//		// Rotate the sample
//
//		// Rotate the scene
//
//		// Update the X-ray image
//		updateXRayImage();
		g_xray_renderer.printXRayImage("out.png");

		// Make fullscreen

	    	// Render here
//	    	display();

	// Return an exit code
	return (EXIT_SUCCESS);
}



//
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
//            g_xray_renderer.printXRayImage("out.png");
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
//
////-------------
//void drawMono()
////-------------
//{
//	draw();
//}
//
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
//
//
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


////-----------------
void loadDetector()
////-----------------
{
	g_xray_detector.setResolutionInUnitsOfLengthPerPixel(g_resolution);
	g_xray_detector.setNumberOfPixels(g_number_of_pixels);
	g_xray_detector.setDetectorPosition(g_detector_position);
	g_xray_detector.setUpVector(g_detector_up_vector);

	// The X-ray image is not up-to-date
	g_is_xray_image_up_to_date = false;
}


////---------------
void loadSource()
////---------------
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


////----------------------
void loadXRaySimulator()
////----------------------
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


////--------------------
//void updateXRayImage()
////--------------------
//{
//	// The X-ray image is not up-to-date
//	if (!g_is_xray_image_up_to_date)
//	{
//		// Compute the X-Ray image
//		g_xray_renderer.computeImage(g_sample_rotation_matrix);
//
//		// Normalise the X-ray image
//		g_xray_renderer.normalise();
//
//		// The X-ray image is up-to-date
//		g_is_xray_image_up_to_date = true;
//	}
//}
//
//
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
//
//
////--------------------------------
//float radian2degree(float anAngle)
////--------------------------------
//{
//	return (180.0 * anAngle / gVirtualXRay::PI);
//}
//
//
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
