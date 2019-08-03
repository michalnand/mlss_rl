%module gl_gui

%include <std_vector.i>
%include <std_string.i>

%apply const std::string& {std::string* foo};



%{
#include <string>
#include <vector>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>

#include <load_textures.h>
#include <json_config.h>
#include <glvisualisation.h>
%}


/*
%include <string>
%include <vector>
%include <jsoncpp/json/json.h>
%include <jsoncpp/json/value.h>
*/

%include <load_textures.h>
%include <json_config.h>
%include <glvisualisation.h>
