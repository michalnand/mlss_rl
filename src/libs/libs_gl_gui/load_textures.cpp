#include "load_textures.h"

#include <vector>
#include <CImg.h>

#include <iostream>
#include <fstream>

LoadTextures::LoadTextures()
{
  textures_count = 0;
}

LoadTextures::LoadTextures(std::string file_name)
{
  load(file_name);
}

LoadTextures::~LoadTextures()
{

}

void LoadTextures::load(std::string file_name)
{
  std::ifstream f(file_name.c_str());

  if (f.good() != true)
  {
    std::cout << "no such texture file " << file_name << "\n";
	
     file_name = "gui/"+file_name;
	  std::ifstream f_alternative(file_name.c_str());

	  if (f_alternative.good() != true)
	  {
	    std::cout << "no such alternative texture file " << file_name << "\n";

	    return;
	  }
  }



  JsonConfig json(file_name);

  textures_count = json.result["textures"].size();

  textures.resize(textures_count);
  map.clear();

  glGenTextures(textures_count, &textures[0]);

  for (unsigned int i = 0; i < textures_count; i++) 
  {
    std::string texture_file_name = json.result["textures"][i]["file_name"].asString();
    unsigned int texture_id = json.result["textures"][i]["id"].asInt();


    map[texture_id] = i;

    cimg_library::CImg<unsigned char> image(texture_file_name.c_str());

    unsigned int width  = image.width();
    unsigned int height = image.height();

    std::vector<unsigned char> image_data;
    image_data.resize(width*height*3);

    unsigned int ptr = 0;
    for (unsigned int y = 0; y < height; y++)
      for (unsigned int x = 0; x < width; x++)
        for (unsigned int ch = 0; ch < 3; ch++)
        {
          image_data[ptr] = *(image.data(x, y, 0, ch));
          ptr++;
        }

    glBindTexture(GL_TEXTURE_2D, textures[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
        GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)(&image_data[0]));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    std::cout << "loading texture ";
    std::cout << texture_id << " ";
    std::cout << textures[i] << " ";
    std::cout << texture_file_name << "\n";
  }
}



unsigned int LoadTextures::get_textures_count()
{
  return textures_count;
}

GLuint LoadTextures::get(unsigned int id)
{
  unsigned int idx = map[id];
  return textures[idx];
}

GLuint LoadTextures::get_idx(unsigned int idx)
{
  return textures[idx];
}
