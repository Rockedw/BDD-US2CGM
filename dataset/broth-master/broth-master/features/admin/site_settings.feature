Feature: Site Settings
  In order to control how pieces of the site function
  An admin
  Should be able to change site settings
  
  Background:
    Given I am signed up and signed in as an admin "admin@person.com/password"
  
  Scenario: Admin turns on beta invites
    Given site settings have been disabled for "beta_invites"
    When I go to the edit admin site settings page
    And I check "Enable beta invites?"
    And I press "Update"
    And I follow "Invites"
    Then I should see "Pending invites"
  
  
  