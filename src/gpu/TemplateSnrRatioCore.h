#ifndef TemplateSnrRatioCore_H_
#define TemplateSnrRatioCore_H_

class TemplateSnrRatioCore {

  public:
    TemplateSnrRatioCore( );
    TemplateSnrRatioCore(int number_of_jobs);
    virtual ~TemplateSnrRatioCore( );

    void Init(int number_of_jobs);

    DeviceManager gpuDev;

    int nGPUs;
    int nThreads;
    int number_of_jobs_per_image_in_gui;

    // CPU images to be passed in -
    Image input_reconstruction_particle, input_reconstruction_correct, input_reconstruction_wrong;
    Image current_projection_image, current_projection_correct_template, current_projection_other;
    Image input_image; // These will be modified on the host from withing Template Matching Core so Allocate locally

    cudaGraph_t     graph;
    cudaGraphExec_t graphExec;
    bool            is_graph_allocated = false;

    // These are assumed to be empty containers at the outset, so xfer host-->device is skipped
    GpuImage d_max_intensity_projection_ac, d_max_intensity_projection_cc;
    GpuImage d_best_psi;
    GpuImage d_best_phi;
    GpuImage d_best_theta;
    GpuImage d_best_defocus;
    GpuImage d_best_pixel_size;

    GpuImage d_sum1_ac, d_sum2_ac, d_sum3_ac, d_sum4_ac, d_sum5_ac;
    GpuImage d_sum1_cc, d_sum2_cc, d_sum3_cc, d_sum4_cc, d_sum5_cc;
    GpuImage d_sumSq1_ac, d_sumSq2_ac, d_sumSq3_ac, d_sumSq4_ac, d_sumSq5_ac;
    GpuImage d_sumSq1_cc, d_sumSq2_cc, d_sumSq3_cc, d_sumSq4_cc, d_sumSq5_cc;

    bool is_allocated_sum_buffer = false;
    int  is_non_zero_sum_buffer;
    // This will need to be copied in
    GpuImage d_input_image;
    GpuImage d_current_projection_image, d_current_projection_correct_template, d_current_projection_other;
    GpuImage d_padded_image, d_padded_reference_correct, d_padded_reference_wrong;

    //GpuImage d_padded_reference;

    // Search range parameters
    float pixel_size;
    float psi_max;
    float psi_start;
    float psi_step_sampled_view;
    float psi_step_tm;

    int current_search_position_sampled_view;
    int current_search_position_tm;

    int number_of_rotations_sampled_view;
    int number_of_rotations_tm;

    int first_search_position_sampled_view;
    int last_search_position_sampled_view;
    int first_search_position_tm;
    int last_search_position_tm;

    long total_correlation_positions_sampled_view, total_correlation_positions_sampled_view_per_thread;

    float avg_for_normalization, std_for_normalization;
    long  total_number_of_cccs_calculated;

    // Search objects
    AnglesAndShifts angles_sampled_view, angles_tm;
    EulerSearch     global_euler_search_sampled_view, global_euler_search_tm;

    ProgressBar* my_progress;

    MyApp* parent_pointer;

    __half2 *my_stats_ac, *my_stats_cc;
    __half2 *my_peaks_ac, *my_peaks_cc;
    __half2 *my_new_peaks_ac, *my_new_peaks_cc; // for passing euler angles to the callback
    void     MipPixelWise(__half psi, __half theta, __half phi, const int view_counter);

    void WriteMipToImage(int view_counter);
    void UpdateSums(__half2* my_stats, GpuImage& sum, GpuImage& sq_sum, const int view_counter);

    void Init(MyApp*           parent_pointer,
              Image&           input_reconstruction_particle,
              Image&           input_reconstruction_correct,
              Image&           input_reconstruction_wrong,
              Image&           input_image,
              Image&           current_projection_image,
              Image&           current_projection_correct_template,
              Image&           current_projection_wrong,
              float            psi_max,
              float            psi_start,
              float            psi_step_sampled_view,
              float            psi_step_tm,
              AnglesAndShifts& angles_sampled_view,
              AnglesAndShifts& angles_tm,
              EulerSearch&     global_euler_search_sampled_view,
              EulerSearch&     global_euler_search_tm,
              int              first_search_position_sampled_view,
              int              last_search_position_sampled_view,
              int              first_search_position_tm,
              int              last_search_position_tm,
              ProgressBar*     my_progress,
              int              number_of_rotations_sampled_view,
              long             total_correlation_positions_sampled_view,
              long             total_correlation_positions_sampled_view_per_thread,
              float            avg_for_normalization,
              float            std_for_normalization);

    void RunInnerLoop(Image& projection_filter, int threadIDX, long& current_correlation_position_sampled_view);

  private:
};

#endif
