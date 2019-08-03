#ifndef _LOAD_TEXTURES_H_
#define _LOAD_TEXTURES_H_

#include <GL/gl.h>
#include <string>
#include <vector>
#include <map>

#include <json_config.h>


class LoadTextures
{
  private:
    std::map<unsigned int, unsigned int> map;
    std::vector<GLuint> textures;
    unsigned int textures_count;

  public:
    LoadTextures();
    LoadTextures(std::string file_name);
    virtual ~LoadTextures();

    void load(std::string file_name);

    unsigned int get_textures_count();

    GLuint get(unsigned int id);
    GLuint get_idx(unsigned int idx);

};


#endif
