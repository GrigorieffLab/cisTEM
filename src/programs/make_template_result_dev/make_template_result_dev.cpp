#include "../../core/core_headers.h"

#include "../../constants/constants.h"
#include <vector> // for std::vector
#include <algorithm> // for std::max
#include <utility> // for std::pair

using namespace std;

class
        MakeTemplateResultDev : public MyApp {
  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(MakeTemplateResultDev)

// override the DoInteractiveUserInput

void MakeTemplateResultDev::DoInteractiveUserInput( ) {

    wxString input_scaled_mip_filename;
    wxString input_mip_filename;
    wxString input_best_psi_filename;
    wxString input_best_theta_filename;
    wxString input_best_phi_filename;
    wxString input_best_defocus_filename;
    wxString input_best_pixel_size_filename;
    wxString xyz_coords_filename;

    int   min_peak_radius;
    float pixel_size;
    int   ignore_N_pixels_from_the_border = -1;

    UserInput* my_input = new UserInput("MakeTemplateResultDev", 1.00);

    input_scaled_mip_filename       = my_input->GetFilenameFromUser("Input scaled MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
    input_mip_filename              = my_input->GetFilenameFromUser("Input MIP file", "The file for saving the maximum intensity projection image", "mip.mrc", false);
    input_best_psi_filename         = my_input->GetFilenameFromUser("Input psi file", "The file containing the best psi image", "psi.mrc", false);
    input_best_theta_filename       = my_input->GetFilenameFromUser("Input theta file", "The file containing the best psi image", "theta.mrc", false);
    input_best_phi_filename         = my_input->GetFilenameFromUser("Input phi file", "The file containing the best psi image", "phi.mrc", false);
    input_best_defocus_filename     = my_input->GetFilenameFromUser("Input defocus file", "The file with the best defocus image", "defocus.mrc", true);
    input_best_pixel_size_filename  = my_input->GetFilenameFromUser("Input pixel size file", "The file with the best pixel size image", "pixel_size.mrc", true);
    xyz_coords_filename             = my_input->GetFilenameFromUser("Output x,y,z coordinate file", "The file for saving the x,y,z coordinates of the found targets", "coordinates.txt", false);
    min_peak_radius                 = my_input->GetIntFromUser("Min Peak Radius (px.)", "Essentially the minimum closeness for peaks", "10", 1);
    pixel_size                      = my_input->GetFloatFromUser("Pixel size of images (A)", "Pixel size of input images in Angstroms", "1.0", 0.0);
    ignore_N_pixels_from_the_border = my_input->GetIntFromUser("Ignore N pixels from the edge of the MIP", "Defaults to 1/2 the template dimension (-1)", "-1", -1);

    delete my_input;

    //	my_current_job.Reset(14);
    my_current_job.ManualSetArguments("ttttttttifi", input_scaled_mip_filename.ToUTF8( ).data( ),
                                      input_mip_filename.ToUTF8( ).data( ),
                                      input_best_psi_filename.ToUTF8( ).data( ),
                                      input_best_theta_filename.ToUTF8( ).data( ),
                                      input_best_phi_filename.ToUTF8( ).data( ),
                                      input_best_defocus_filename.ToUTF8( ).data( ),
                                      input_best_pixel_size_filename.ToUTF8( ).data( ),
                                      xyz_coords_filename.ToUTF8( ).data( ),
                                      min_peak_radius,
                                      pixel_size,
                                      ignore_N_pixels_from_the_border);
}

// override the do calculation method which will be what is actually run..

bool MakeTemplateResultDev::DoCalculation( ) {

    wxDateTime start_time = wxDateTime::Now( );

    wxString input_scaled_mip_filename       = my_current_job.arguments[0].ReturnStringArgument( );
    wxString input_mip_filename              = my_current_job.arguments[1].ReturnStringArgument( );
    wxString input_best_psi_filename         = my_current_job.arguments[2].ReturnStringArgument( );
    wxString input_best_theta_filename       = my_current_job.arguments[3].ReturnStringArgument( );
    wxString input_best_phi_filename         = my_current_job.arguments[4].ReturnStringArgument( );
    wxString input_best_defocus_filename     = my_current_job.arguments[5].ReturnStringArgument( );
    wxString input_best_pixel_size_filename  = my_current_job.arguments[6].ReturnStringArgument( );
    wxString xyz_coords_filename             = my_current_job.arguments[7].ReturnStringArgument( );
    int      min_peak_radius                 = my_current_job.arguments[8].ReturnIntegerArgument( );
    float    pixel_size                      = my_current_job.arguments[9].ReturnFloatArgument( );
    int      ignore_N_pixels_from_the_border = my_current_job.arguments[10].ReturnIntegerArgument( );

    Image scaled_mip_image;
    Image mip_image;
    Image psi_image;
    Image theta_image;
    Image phi_image;
    Image defocus_image;
    Image pixel_size_image;
    int   result_number = 1;
    Peak  current_peak;

    float current_phi;
    float current_theta;
    float current_psi;
    float current_defocus;
    float current_pixel_size;

    int   number_of_peaks_found = 0;
    float sq_dist_x, sq_dist_y;
    long  address;
    long  text_file_access_type;

    float                          coordinates[8];
    vector<tuple<float, int, int>> localMaxima; // Peak, Row, Column

    text_file_access_type = OPEN_TO_WRITE;
    NumericTextFile coordinate_file(xyz_coords_filename, text_file_access_type, 8);
    coordinate_file.WriteCommentLine("         Psi          Theta            Phi              X              Y              Z      PixelSize           Peak");

    mip_image.QuickAndDirtyReadSlice(input_mip_filename.ToStdString( ), result_number);
    scaled_mip_image.QuickAndDirtyReadSlice(input_scaled_mip_filename.ToStdString( ), result_number);
    psi_image.QuickAndDirtyReadSlice(input_best_psi_filename.ToStdString( ), result_number);
    theta_image.QuickAndDirtyReadSlice(input_best_theta_filename.ToStdString( ), result_number);
    phi_image.QuickAndDirtyReadSlice(input_best_phi_filename.ToStdString( ), result_number);
    defocus_image.QuickAndDirtyReadSlice(input_best_defocus_filename.ToStdString( ), result_number);
    pixel_size_image.QuickAndDirtyReadSlice(input_best_pixel_size_filename.ToStdString( ), result_number);
    int mip_x_dimension = mip_image.logical_x_dimension;
    int mip_y_dimension = mip_image.logical_y_dimension;

    if ( ignore_N_pixels_from_the_border > 0 && (ignore_N_pixels_from_the_border > mip_image.logical_x_dimension / 2 || ignore_N_pixels_from_the_border > mip_image.logical_y_dimension / 2) ) {
        wxPrintf("You have entered %d for ignore_N_pixels_from_the_border, which is too large given image half dimesnsions of %d (X) and %d (Y)",
                 ignore_N_pixels_from_the_border, mip_x_dimension / 2, mip_y_dimension / 2);
        exit(-1);
    }

    wxPrintf("\n");

    // int pixel_counter = 0;
    for ( int i = ignore_N_pixels_from_the_border; i <= mip_x_dimension - ignore_N_pixels_from_the_border; i++ ) {
        for ( int j = ignore_N_pixels_from_the_border; j <= mip_y_dimension - ignore_N_pixels_from_the_border; j++ ) {

            float maxVal = scaled_mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0);

            bool isLocalMax = true;
            // Check the neighborhood
            for ( int ki = -min_peak_radius; ki <= min_peak_radius; ++ki ) {
                for ( int kj = -min_peak_radius; kj <= min_peak_radius; ++kj ) {
                    int ni = i + ki; // neighbor row index
                    int nj = j + kj; // neighbor column index
                    // Check boundaries and find maximum
                    if ( ni >= 0 && ni < mip_x_dimension && nj >= 0 && nj < mip_y_dimension ) {

                        if ( scaled_mip_image.ReturnRealPixelFromPhysicalCoord(ni, nj, 0) > maxVal ) {
                            maxVal     = scaled_mip_image.ReturnRealPixelFromPhysicalCoord(ni, nj, 0); // CHECKME: should ni,nj be logical or physical coordinates?
                            isLocalMax = false;
                        }
                    }
                }
            }

            if ( isLocalMax && maxVal == scaled_mip_image.ReturnRealPixelFromPhysicalCoord(i, j, 0) ) {
                localMaxima.emplace_back(maxVal, i, j);
                // write into output coordinates

                number_of_peaks_found++;
            }

            //pixel_counter++;
        }
        //pixel_counter += scaled_mip_image.padding_jump_value;
    }

    // Sort based on the local maxima values (descending order)
    sort(localMaxima.begin( ), localMaxima.end( ), [](const tuple<float, int, int>& a, const tuple<float, int, int>& b) {
        return get<0>(a) > get<0>(b); // Sorting by first element of tuple, which is the max value
    });

    int coord_x, coord_y;
    for ( const auto& entry : localMaxima ) {
        coord_x = get<1>(entry);
        coord_y = get<2>(entry);

        coordinates[0] = psi_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
        coordinates[1] = theta_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
        coordinates[2] = phi_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
        coordinates[3] = coord_x * pixel_size;
        coordinates[4] = coord_y * pixel_size;
        coordinates[5] = defocus_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
        coordinates[6] = pixel_size_image.ReturnRealPixelFromPhysicalCoord(coord_x, coord_y, 0);
        coordinates[7] = get<0>(entry);
        coordinate_file.WriteLine(coordinates);
    }

    if ( is_running_locally == true ) {
        wxPrintf("\nFound %i peaks.\n\n", number_of_peaks_found);
        wxPrintf("\nMake Template Results: Normal termination\n");
        wxDateTime finish_time = wxDateTime::Now( );
        wxPrintf("Total Run Time : %s\n\n", finish_time.Subtract(start_time).Format("%Hh:%Mm:%Ss"));
    }

    return true;
}
