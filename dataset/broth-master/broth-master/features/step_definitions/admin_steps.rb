Given /^I am signed up and signed in as an admin "(.*)\/(.*)"$/ do |email, password|
  user = Factory :admin_user,
    :email                 => email,
    :password              => password,
    :password_confirmation => password
  When %{I sign in as "#{email}\/#{password}"}
end